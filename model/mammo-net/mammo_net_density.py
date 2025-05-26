import argparse
import cv2
import copy
import csv
import numbers
import numpy as np
import os
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from argparse import ArgumentParser, Namespace
from data_utils import *
from get_embed_csv import EmbedCSVGenerator, EmbedDensityTestCSVBuilder, TestType
from sklearn.utils import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from torchmetrics.functional import auroc
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import CenterCrop
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from stocaching import SharedCache
from sampler import SamplerFactory

image_size = (224, 168)     # image input size
test_percent = 0.2          # how much of total samples are used for testing (default 20%)
val_percent = 0.2           # how much of total training samples are used for model selection (default 20%)
num_classes = 2

class GammaCorrectionTransform:
    """Apply Gamma Correction to the image"""
    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, 'gammacorrection')   
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for gamma correction do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: gamma corrected image.
        """
        gamma_factor = None if self.gamma is None else float(torch.empty(1).uniform_(self.gamma[0], self.gamma[1]))
        if gamma_factor is not None:
            img = TF.adjust_gamma(img, gamma_factor, gain=1)
        return img

class MammoDataset(Dataset):
    def __init__(self, data, image_size, image_normalization, horizontal_flip = False, augmentation = False, cache_size = 0, test_set = False):
        self.image_size = image_size
        self.image_normalization = image_normalization
        self.do_flip = horizontal_flip
        self.do_augment = augmentation
        self.center_crop = CenterCrop(self.image_size)
        self.test_set = test_set

        # photometric data augmentation
        self.photometric_augment = T.Compose([
            GammaCorrectionTransform(gamma=0.2),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # geometric data augmentation
        self.geometric_augment = T.Compose([
            # T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=10, scale=(0.9, 1.1))], p=0.5),
        ])

        self.img_paths = data.img_path.to_numpy()
        self.study_ids = data.study_id.to_numpy()
        self.image_ids = data.image_id.to_numpy()
        self.labels = data.density_label.to_numpy()
        self.laterality = data.laterality.to_numpy()

        # initialize the cache
        self.cache = None
        self.use_cache = cache_size > 0
        if self.use_cache:
            self.cache = SharedCache(
                size_limit_gib=cache_size,
                dataset_len=self.labels.shape[0],
                data_dims=(1, image_size[0], image_size[1]),
                dtype=torch.float32,
            )

    def preprocess(self, image, horizontal_flip):

        # resample
        if self.image_size != image.shape:
            image = resize(image, output_shape=self.image_size, preserve_range=True)
        
        # breast mask
        image_norm = image - np.min(image)
        image_norm = image_norm / np.max(image_norm)
        thresh = cv2.threshold(img_as_ubyte(image_norm), 5, 255, cv2.THRESH_BINARY)[1]

        # Connected components with stats.
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)

        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        max_label, _ = max(
            [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
            key=lambda x: x[1],
        )
        mask = output == max_label
        image[mask == 0] = 0
        
        # flip
        if horizontal_flip:
            l = np.mean(image[:,0:int(image.shape[1]/2)])
            r = np.mean(image[:,int(image.shape[1]/2)::])
            if l < r:
                image = image[:, ::-1].copy()
        
        return image

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        image = None
        if self.use_cache:
            image = self.cache.get_slot(index)
        
        if image is None:
            img_path = self.img_paths[index]
            image = imread(img_path).astype(np.float32)
            horizontal_flip = self.do_flip #and self.laterality[index] == 'R'
            image = self.preprocess(image, horizontal_flip)
            image = torch.from_numpy(image).unsqueeze(0)            
            
            if self.use_cache:
                self.cache.set_slot(index, image, allow_overwrite=True)
        
        if self.test_set:
            # apply center crop and rescale images from [-1, 1] to [0, 1]
            tsfm = CenterCrop((224,168))
            image = tsfm((image + 1) / 2)  

        # normalize intensities to range [0,1]
        image = image / self.image_normalization

        if self.do_augment:
            image = self.photometric_augment(image)
            image = self.geometric_augment(image)

        image = image.repeat(3, 1, 1)

        return {'image': image, 'label': self.labels[index], 'study_id': self.study_ids[index], 'image_id': self.image_ids[index]}
    
    def get_labels(self):
        return self.labels
    
# TODO: consider moving, train_data_size test_data_size

class  EMBEDMammoDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir, 
                 image_size, 
                 test_percent, 
                 val_percent, 
                 batch_alpha, 
                 batch_size, 
                 num_workers, 
                 use_counterfactuals, 
                 test_type=TestType.BALANCED, 
                 train_data_size=None, 
                 test_data_size=1000):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.test_percent = test_percent
        self.val_percent = val_percent
        self.batch_alpha = batch_alpha
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_counterfactuals = use_counterfactuals
        self.max_data_size = train_data_size

        train_data, test_data = self._get_data_from_csv(use_counterfactuals, train_data_size, test_data_size, test_type)
        self.train_data = self._prepare_dataframe(df=train_data, data_dir=embed_counterfactuals_dir)
        self.test_data = self._prepare_dataframe(df=test_data, data_dir=embed_data_dir)

        # Split train_data into training and validation
        # Making sure images from the same subject are within the same set
        unique_study_ids_train = self.train_data.empi_anon.unique()
        unique_study_ids_train = shuffle(unique_study_ids_train)

        num_train = (round(len(unique_study_ids_train) * (1.0 - self.val_percent)))
        val_sub_id = unique_study_ids_train[num_train:]

        self.train_data['split'] = 'training'
        self.train_data['split'] = 'training'
        self.train_data.loc[self.train_data.empi_anon.isin(val_sub_id), 'split'] = 'validation'

        train_df = self.train_data[self.train_data['split'] == 'training']
        val_df = self.train_data[self.train_data['split'] == 'validation']

        # Dataset objects
        self.train_set = MammoDataset(train_df, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=True, cache_size=4, test_set=False)
        self.val_set = MammoDataset(val_df, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=False, test_set=False)
        self.test_set = MammoDataset(self.test_data, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=False, test_set=True)

        train_labels = self.train_set.get_labels()
        train_class_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])

        val_labels = self.val_set.get_labels()
        val_class_count = np.array([len(np.where(val_labels == t)[0]) for t in np.unique(val_labels)])

        test_labels = self.test_set.get_labels()
        test_class_count = np.array([len(np.where(test_labels == t)[0]) for t in np.unique(test_labels)])

        if self.batch_alpha > 0:
            train_class_idx = [np.where(train_labels == t)[0] for t in np.unique(train_labels)]
            train_batches = len(self.train_set) // self.batch_size            

            self.train_sampler = SamplerFactory().get(
                    train_class_idx,
                    self.batch_size,
                    train_batches,
                    alpha=self.batch_alpha,
                    kind='fixed',
                )

        print('samples (train): ',len(self.train_set))
        print('samples (val):   ',len(self.val_set))
        print('samples (test):  ',len(self.test_set))
        print(train_class_count)
        print(val_class_count)
        print(test_class_count)

    def _get_data_from_csv(self,use_counterfactuals, train_data_size, test_data_size, test_type):
        train_csv_generator = EmbedCSVGenerator(use_counterfactuals=use_counterfactuals, train_data_size=train_data_size)
        train_data = train_csv_generator.get_train_csv()

        test_csv_generator = EmbedDensityTestCSVBuilder(test_data_size=test_data_size)
        test_data = test_csv_generator.get_test_csv(test_type)

        return train_data, test_data

    def _prepare_dataframe(self, df, data_dir):
        # FFDM only
        df = df[df['FinalImageType'] == '2D']

        df = df[df['tissueden'].notna()]
        df = df[df['tissueden'] < 5]

        # MLO and CC only
        df = df[df['ViewPosition'].isin(['MLO', 'CC'])]

        # Update paths and IDs
        df['img_path'] = [os.path.join(data_dir, p) for p in df.image_path.values]
        df['study_id'] = df.empi_anon.astype(str)

        # Set image ID based on counterfactuals
        df['image_id'] = (
            [Path(img_path).name for img_path in df.image_path.values]
            if not self.use_counterfactuals
            else ['/'.join(str(img_path).split('/')[-2:]) for img_path in df.image_path.values]
        )

        df['laterality'] = df['ImageLateralityFinal']

        # Density label
        df['density_label'] = 0
        if num_classes == 4:
            df.loc[df['tissueden'] == 1, 'density_label'] = 0
            df.loc[df['tissueden'] == 2, 'density_label'] = 1
            df.loc[df['tissueden'] == 3, 'density_label'] = 2
            df.loc[df['tissueden'] == 4, 'density_label'] = 3
        elif num_classes == 2:
            df.loc[df['tissueden'].isin([1, 2]), 'density_label'] = 0
            df.loc[df['tissueden'].isin([3, 4]), 'density_label'] = 1

        return df
    
    def train_dataloader(self):
        if self.batch_alpha == 0:
            return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(dataset=self.train_set, batch_sampler=self.train_sampler, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class MammoNet(pl.LightningModule):
    def __init__(self, backbone='resnet18', learning_rate=0.0001, checkpoint=None):
        super().__init__()
        self.num_classes = num_classes
        self.lr = learning_rate
        self.backbone = backbone

        # Default model is a ResNet-18 pre-trained on ImageNet
        if self.backbone == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif self.backbone == 'resnet34':
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif self.backbone == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        if checkpoint is not None:
            print(self.model.load_state_dict(state_dict=checkpoint, strict=False))                

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def process_batch(self, batch):
        img, lab = batch['image'], batch['label']
        out = self.forward(img)
        prd = torch.softmax(out, dim=1)
        loss = F.cross_entropy(out, lab) 
        return loss, prd, lab

    def on_train_epoch_start(self):
        self.train_preds = []
        self.train_trgts = []

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.log('train_loss', loss, batch_size=lab.shape[0])
        self.train_preds.append(prd.detach().cpu())
        self.train_trgts.append(lab.detach().cpu())
        # batch_pos = len(np.where(lab.detach().cpu().numpy() == 1)[0])
        # batch_neg = len(np.where(lab.detach().cpu().numpy() == 0)[0])
        # self.log('batch_pos_count', batch_pos)
        # self.log('batch_pos_percent', batch_pos / (batch_pos + batch_neg) * 100.0)
        if batch_idx == 0:
            images = batch['image'][0:4, ...].detach().cpu()
            grid = torchvision.utils.make_grid(images, nrow=2, normalize=True)
            self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def on_train_epoch_end(self):
        self.train_preds = torch.cat(self.train_preds, dim=0)
        self.train_trgts = torch.cat(self.train_trgts, dim=0)
        auc = auroc(self.train_preds, self.train_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('train_auc', auc)
        self.train_preds = []
        self.train_trgts = []

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_trgts = []

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.log('val_loss', loss, batch_size=lab.shape[0])
        self.val_preds.append(prd.detach().cpu())
        self.val_trgts.append(lab.detach().cpu())

    def on_validation_epoch_end(self):
        self.val_preds = torch.cat(self.val_preds, dim=0)
        self.val_trgts = torch.cat(self.val_trgts, dim=0)
        auc = auroc(self.val_preds, self.val_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('val_auc', auc)
        self.val_preds = []
        self.val_trgts = []

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_trgts = []
        self.test_study_ids = []
        self.test_image_ids = []

    def test_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.log('test_loss', loss, batch_size=lab.shape[0])
        self.test_preds.append(prd.detach().cpu())
        self.test_trgts.append(lab.detach().cpu())        
        self.test_study_ids.append(batch['study_id'])
        self.test_image_ids.append(batch['image_id'])


    def on_test_epoch_end(self):
        self.test_preds = torch.cat(self.test_preds, dim=0)
        self.test_trgts = torch.cat(self.test_trgts, dim=0)
        auc = auroc(self.test_preds, self.test_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('test_auc', auc)


class MammoNetEmbeddings(MammoNet):
    def __init__(self, init=None):
        super().__init__()
        self.embeddings = []
        if init is not None:
            self.model = init.model

    def on_test_start(self):
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity(num_features)
        self.embeddings = []
        self.test_study_ids = []
        self.test_image_ids = []

    def test_step(self, batch, batch_idx):
        emb = self.forward(batch['image'])
        self.embeddings.append(emb.detach().cpu())
        self.test_study_ids.append(batch['study_id'])
        self.test_image_ids.append(batch['image_id'])

    def on_test_epoch_end(self):
        self.embeddings = torch.cat(self.embeddings, dim=0)


def save_predictions(model, output_fname):
    std_ids = [id for sublist in model.test_study_ids for id in sublist]
    img_ids = [id for sublist in model.test_image_ids for id in sublist]
    cols_names = ['class_' + str(i) for i in range(0, num_classes)]
    df = pd.DataFrame(data=model.test_preds.numpy(), columns=cols_names)    
    df['target'] = model.test_trgts.numpy()
    df['study_id'] = std_ids
    df['image_id'] = img_ids
    df.to_csv(output_fname, index=False)


def save_embeddings(model, output_fname):
    std_ids = [id for sublist in model.test_study_ids for id in sublist]
    img_ids = [id for sublist in model.test_image_ids for id in sublist]
    df = pd.DataFrame(data=model.embeddings.numpy())
    df['study_id'] = std_ids
    df['image_id'] = img_ids
    df.to_csv(output_fname, index=False)

def extract_hparams_from_foldername(name):
    name = name.strip()
    pattern = r'cf(?P<cf>\w+)_bs(?P<bs>\d+)_lr(?P<lr>[\d.]+)_epochs(?P<epochs>\d+)'
    match = re.match(pattern, name)

    if not match:
        print('checkpoint folder name does not match expected pattern')
        return None
    cf, bs, lr, epochs = match.groups()
    return {
        'cf': cf == 'True',
        'model': 'resnet50',
        'batch_size': int(bs),
        'learning_rate': float(lr),
        'epochs': int(epochs),
        'seed': 42,
        'num_devices': 1,
        'num_workers': 6,
        'batch_alpha': 1.0
    }

def main(hparams):

    # torch.set_float32_matmul_precision('medium')
    torch.set_float32_matmul_precision('high')
                                       
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(hparams.seed, workers=True)

    if hparams.dataset == 'embed':
        data = EMBEDMammoDataModule(data_dir=embed_counterfactuals_dir,
                                image_size=image_size,
                                test_percent=test_percent,
                                val_percent=val_percent,
                                batch_alpha=hparams.batch_alpha,
                                batch_size=hparams.batch_size,
                                num_workers=hparams.num_workers, 
                                use_counterfactuals=hparams.counterfactuals,
                                train_data_size=hparams.max_data_size)
    else:
        print('Unknown dataset. Exiting.')
        return

    # model
    model = MammoNet(backbone=hparams.model, learning_rate=hparams.learning_rate)
    
    # Create output directory
    output_dir = os.path.join(hparams.output_root,hparams.output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('')
    print('=============================================================')
    print('TRAINING...')
    print('=============================================================')
    print('')
    
    early_stop_callback = EarlyStopping(
        monitor='val_auc',       
        mode='max',              
        patience=5,              # stop after 5 epochs without improvement
        verbose=True
    )

    # train
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        accelerator='auto',
        devices=hparams.num_devices,
        precision='16-mixed',
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(hparams.output_root, name=hparams.output_name),
        callbacks=[
            ModelCheckpoint(monitor='val_auc', mode='max'), 
            early_stop_callback,
            TQDMProgressBar(refresh_rate=10) 
            ],
    )
    trainer.fit(model=model, datamodule=data)
    
    print('')
    print('=============================================================')
    print('VALIDATION...')
    print('=============================================================')
    print('')

    trainer.validate(model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)

    print('')
    print('=============================================================')
    print('TESTING...')
    print('=============================================================')
    print('')

    trainer.test(model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)
    save_predictions(model=model, output_fname=os.path.join(output_dir, 'predictions.csv'))

    print('')
    print('=============================================================')
    print('EMBEDDINGS...')
    print('=============================================================')
    print('')

    # model_modified = MammoNetEmbeddings(init=model)
    # trainer.test(model=model_modified, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)
    # save_embeddings(model=model, output_fname=os.path.join(output_dir, 'embeddings.csv'))

def run_hyperparameter_sweep():
    epochs = [5]
    batches_lrs = [(128, 3e-4), (256, 5e-4)]
    cfs = [True, False]

    base_args = Namespace(
        epochs=10,
        batch_size=32,
        batch_alpha=1.0,
        learning_rate=0.0001,
        num_workers=6,
        num_devices=1,
        model='resnet50',
        dataset='embed',
        csv_file='data/embed-non-negative.csv',
        output_root='please_work',
        output_name='debug',
        seed=42,
        counterfactuals=False,
        max_data_size=None
    )

    for batch_size, lr in batches_lrs:
        for num_epochs in epochs:
                for cf in cfs:
                    output_name = f"cf{cf}_bs{batch_size}_lr{lr}_epochs{num_epochs}"
                    output_path = os.path.join(base_args.output_root, output_name)

                    if os.path.exists(output_path):
                        print(f"skipping existing run: {output_name}")
                        continue

                    args = copy.deepcopy(base_args)
                    args.epochs = num_epochs
                    args.batch_size = batch_size
                    args.learning_rate = lr
                    args.counterfactuals = cf

                    args.output_name = output_name

                    print(f"Running with: {args.output_name}")
                    main(args)

def train_model():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_alpha', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='embed')
    parser.add_argument('--csv_file', type=str, default='data/embed-non-negative.csv')
    parser.add_argument('--output_root', type=str, default='output')
    parser.add_argument('--output_name', type=str, default='debug')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--counterfactuals', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_data_size', type=int, default=None)
    args = parser.parse_args()

    main(args)

# TODO: delete this
def test():

    for i in range(1, 10):
        args = Namespace(
            epochs=i,
            batch_size=256,
            batch_alpha=1.0,
            learning_rate=0.0003,
            num_workers=6,
            num_devices=1,
            model='resnet18',
            dataset='embed',
            csv_file='data/embed-non-negative.csv',
            output_root='sweep_test',
            output_name='debug',
            seed=42,
            counterfactuals=False,
            max_data_size=None
        )
        main(args)


if __name__ == '__main__':
    # run_hyperparameter_sweep()
    # train_model()
    # test_existing_models()
    test()
    
