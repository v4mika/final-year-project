import os
import cv2
import numbers
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from argparse import ArgumentParser
from sklearn.utils import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from torchmetrics.functional import auroc
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from stocaching import SharedCache

from sampler import SamplerFactory

vindr_data_dir = '/vol/biomedic3/data/VinDR-Mammo'
embed_data_dir = '/vol/biomedic3/data/EMBED/images/png/1024x768'
rsna_data_dir = '/vol/biodata/data/Mammo/RSNA/pngs16/1024x768'

# vindr_data_dir = '/data/VinDR-Mammo'      # adjust if data is stored locally
# embed_data_dir = '/data/EMBED/1024x768'   # adjust if data is stored locally

image_size = (1024,768)     # image input size
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
    def __init__(self, data, image_size, image_normalization, horizontal_flip = False, augmentation = False, cache_size = 0):
        self.image_size = image_size
        self.image_normalization = image_normalization
        self.do_flip = horizontal_flip
        self.do_augment = augmentation

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
        self.labels = data.is_positive.to_numpy()
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

        # normalize intensities to range [0,1]
        image = image / self.image_normalization

        if self.do_augment:
            image = self.photometric_augment(image)
            image = self.geometric_augment(image)

        image = image.repeat(3, 1, 1)

        return {'image': image, 'label': self.labels[index], 'study_id': self.study_ids[index], 'image_id': self.image_ids[index]}
    
    def get_labels(self):
        return self.labels


class VinDrMammoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, csv_file, image_size, val_percent, batch_alpha, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.val_percent = val_percent
        self.batch_alpha = batch_alpha
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data = pd.read_csv(csv_file)

        self.data['img_path'] = [os.path.join(self.data_dir, 'pngs', self.data.study_id.values[idx], self.data.image_id.values[idx] + '.png') for idx in range(0, len(self.data))]

        # Define negatives and positives based on BI-RADS categories
        self.data['is_positive'] = self.data['breast_birads']
        self.data.loc[self.data['is_positive'] == 'BI-RADS 1', 'is_positive'] = 0
        self.data.loc[self.data['is_positive'] == 'BI-RADS 2', 'is_positive'] = 0
        self.data.loc[self.data['is_positive'] == 'BI-RADS 3', 'is_positive'] = 0
        self.data.loc[self.data['is_positive'] == 'BI-RADS 4', 'is_positive'] = 0
        self.data.loc[self.data['is_positive'] == 'BI-RADS 5', 'is_positive'] = 1

        # Use pre-defined splits to separate data into development and testing
        self.dev_data = self.data[self.data['split'] == 'training']
        self.test_data = self.data[self.data['split'] == 'test']

        # Split development data into training and validation (for model selection)
        # Making sure images from the same subject are within the same set
        unique_study_ids = self.dev_data.study_id.unique()

        unique_study_ids = shuffle(unique_study_ids)
        num_train = (round(len(unique_study_ids) * (1.0 - self.val_percent)))

        valid_sub_id = unique_study_ids[num_train:]
        self.dev_data.loc[self.dev_data.study_id.isin(valid_sub_id), 'split'] = 'validation'
        
        self.train_data = self.dev_data[self.dev_data['split'] == 'training']
        self.val_data = self.dev_data[self.dev_data['split'] == 'validation']

        self.train_set = MammoDataset(self.train_data, self.image_size, image_normalization=255.0, horizontal_flip=True, augmentation=True, cache_size=48)
        self.val_set = MammoDataset(self.val_data, self.image_size, image_normalization=255.0, horizontal_flip=True, augmentation=False)
        self.test_set = MammoDataset(self.test_data, self.image_size, image_normalization=255.0, horizontal_flip=True, augmentation=False)

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
        print('pos/neg (train): {}/{}'.format(train_class_count[1], train_class_count[0]))
        print('pos/neg (val):   {}/{}'.format(val_class_count[1], val_class_count[0]))
        print('pos/neg (test):  {}/{}'.format(test_class_count[1], test_class_count[0]))
        print('pos (train):     {:0.2f}%'.format(train_class_count[1]/len(train_labels)*100.0))
        print('pos (val):       {:0.2f}%'.format(val_class_count[1]/len(val_labels)*100.0))
        print('pos (test):      {:0.2f}%'.format(test_class_count[1]/len(test_labels)*100.0))

    def train_dataloader(self):
        if self.batch_alpha == 0:
            return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(dataset=self.train_set, batch_sampler=self.train_sampler, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class EMBEDMammoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, csv_file, image_size, test_percent, val_percent, batch_alpha, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.test_percent = test_percent
        self.val_percent = val_percent
        self.batch_alpha = batch_alpha
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data = pd.read_csv(csv_file)
        
        # Screening only
        # self.data = self.data.loc[self.data.desc.str.contains("screen", case=False)].copy()

        # FFDM only
        self.data = self.data[self.data['FinalImageType'] == '2D']

        # Female only
        self.data = self.data[self.data['GENDER_DESC'] == 'Female']

        # Remove unclear breast density cases
        self.data = self.data[self.data['tissueden'].notna()]
        self.data = self.data[self.data['tissueden'] < 5]

        # MLO and CC only        
        self.data = self.data[self.data['ViewPosition'].isin(['MLO','CC'])]

        # Remove spot compression or magnificiation
        # self.data = self.data[self.data['spot_mag'].isna()]

        # Single scanner
        # self.data = self.data[self.data['ManufacturerModelName'] == 'Selenia Dimensions']

        # Single view
        # self.data = self.data[self.data['ViewPosition'] == 'CC']

        self.data['img_path'] = [os.path.join(self.data_dir, img_path) for img_path in self.data.image_path.values]
        self.data['study_id'] = [str(study_id) for study_id in self.data.empi_anon.values]
        self.data['image_id'] = [img_path.split('/')[-1] for img_path in self.data.image_path.values]
        self.data['laterality'] = self.data['ImageLateralityFinal']

        # Split data into training, validation, and testing
        # Making sure images from the same subject are within the same set
        self.data['split'] = 'test'
        
        unique_study_ids_all = self.data.empi_anon.unique()
        unique_study_ids_all = shuffle(unique_study_ids_all)
        num_test = (round(len(unique_study_ids_all) * self.test_percent))
        
        dev_sub_id = unique_study_ids_all[num_test:]
        self.data.loc[self.data.empi_anon.isin(dev_sub_id), 'split'] = 'training'
        
        self.dev_data = self.data[self.data['split'] == 'training']
        self.test_data = self.data[self.data['split'] == 'test']        

        unique_study_ids_dev = self.dev_data.empi_anon.unique()

        unique_study_ids_dev = shuffle(unique_study_ids_dev)
        num_train = (round(len(unique_study_ids_dev) * (1.0 - self.val_percent)))

        valid_sub_id = unique_study_ids_dev[num_train:]
        self.dev_data.loc[self.dev_data.empi_anon.isin(valid_sub_id), 'split'] = 'validation'
        
        self.train_data = self.dev_data[self.dev_data['split'] == 'training']
        self.val_data = self.dev_data[self.dev_data['split'] == 'validation']


        # filtering out non-counterfactuals from testing
        self.test_data = self.test_data['trueclass' in self.test_data['image_path']]

        self.train_set = MammoDataset(self.train_data, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=True, cache_size=48)
        self.val_set = MammoDataset(self.val_data, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=False)
        self.test_set = MammoDataset(self.test_data, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=False)

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
        print('pos/neg (train): {}/{}'.format(train_class_count[1], train_class_count[0]))
        print('pos/neg (val):   {}/{}'.format(val_class_count[1], val_class_count[0]))
        print('pos/neg (test):  {}/{}'.format(test_class_count[1], test_class_count[0]))
        print('pos (train):     {:0.2f}%'.format(train_class_count[1]/len(train_labels)*100.0))
        print('pos (val):       {:0.2f}%'.format(val_class_count[1]/len(val_labels)*100.0))
        print('pos (test):      {:0.2f}%'.format(test_class_count[1]/len(test_labels)*100.0))
    
    def train_dataloader(self):
        if self.batch_alpha == 0:
            return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(dataset=self.train_set, batch_sampler=self.train_sampler, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class RSNAMammoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, csv_file, image_size, test_percent, val_percent, batch_alpha, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.test_percent = test_percent
        self.val_percent = val_percent
        self.batch_alpha = batch_alpha
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data = pd.read_csv(csv_file)
        
        # MLO and CC only        
        self.data = self.data[self.data['view'].isin(['MLO','CC'])]

        self.data['img_path'] = [os.path.join(self.data_dir, str(self.data.patient_id.values[idx]), str(self.data.image_id.values[idx]) + '.png') for idx in range(0, len(self.data))]
        self.data['study_id'] = [str(study_id) for study_id in self.data.patient_id.values]
        self.data['image_id'] = [str(image_id) for image_id in self.data.image_id.values]

        self.data['is_positive'] = self.data['cancer']
        
        # Split data into training, validation, and testing
        # Making sure images from the same subject are within the same set
        self.data['split'] = 'test'
        
        unique_study_ids_all = self.data.study_id.unique()
        unique_study_ids_all = shuffle(unique_study_ids_all)
        num_test = (round(len(unique_study_ids_all) * self.test_percent))
        
        dev_sub_id = unique_study_ids_all[num_test:]
        self.data.loc[self.data.study_id.isin(dev_sub_id), 'split'] = 'training'
        
        self.dev_data = self.data[self.data['split'] == 'training']
        self.test_data = self.data[self.data['split'] == 'test']        

        unique_study_ids_dev = self.dev_data.study_id.unique()

        unique_study_ids_dev = shuffle(unique_study_ids_dev)
        num_train = (round(len(unique_study_ids_dev) * (1.0 - self.val_percent)))

        valid_sub_id = unique_study_ids_dev[num_train:]
        self.dev_data.loc[self.dev_data.study_id.isin(valid_sub_id), 'split'] = 'validation'
        
        self.train_data = self.dev_data[self.dev_data['split'] == 'training']
        self.val_data = self.dev_data[self.dev_data['split'] == 'validation']

        self.train_set = MammoDataset(self.train_data, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=True, cache_size=48)
        self.val_set = MammoDataset(self.val_data, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=False)
        self.test_set = MammoDataset(self.test_data, self.image_size, image_normalization=65535.0, horizontal_flip=True, augmentation=False)

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
        print('pos/neg (train): {}/{}'.format(train_class_count[1], train_class_count[0]))
        print('pos/neg (val):   {}/{}'.format(val_class_count[1], val_class_count[0]))
        print('pos/neg (test):  {}/{}'.format(test_class_count[1], test_class_count[0]))
        print('pos (train):     {:0.2f}%'.format(train_class_count[1]/len(train_labels)*100.0))
        print('pos (val):       {:0.2f}%'.format(val_class_count[1]/len(val_labels)*100.0))
        print('pos (test):      {:0.2f}%'.format(test_class_count[1]/len(test_labels)*100.0))
    
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
        batch_pos = len(np.where(lab.detach().cpu().numpy() == 1)[0])
        batch_neg = len(np.where(lab.detach().cpu().numpy() == 0)[0])
        self.log('batch_pos_count', batch_pos)
        self.log('batch_pos_percent', batch_pos / (batch_pos + batch_neg) * 100.0)
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

    @classmethod
    def from_checkpoint_file(cls, path, **kwargs):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        return cls(checkpoint=state_dict, **kwargs)



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

    def test_step(self, batch, batch_idx):
        emb = self.forward(batch['image'])
        self.embeddings.append(emb.detach().cpu())

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
    df = pd.DataFrame(data=model.embeddings.numpy())
    df.to_csv(output_fname, index=False)


def main(hparams):

    # torch.set_float32_matmul_precision('medium')
    torch.set_float32_matmul_precision('high')
                                       
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(hparams.seed, workers=True)

    # data
    if hparams.dataset == 'vindr':
        data = VinDrMammoDataModule(data_dir=vindr_data_dir,
                                csv_file=hparams.csv_file,
                                image_size=image_size,
                                val_percent=val_percent,
                                batch_alpha=hparams.batch_alpha,
                                batch_size=hparams.batch_size,
                                num_workers=hparams.num_workers)
    elif hparams.dataset == 'embed':
        data = EMBEDMammoDataModule(data_dir=embed_data_dir,
                                csv_file=hparams.csv_file,
                                image_size=image_size,
                                test_percent=test_percent,
                                val_percent=val_percent,
                                batch_alpha=hparams.batch_alpha,
                                batch_size=hparams.batch_size,
                                num_workers=hparams.num_workers)
    elif hparams.dataset == 'rsna':
        data = RSNAMammoDataModule(data_dir=rsna_data_dir,
                                csv_file=hparams.csv_file,
                                image_size=image_size,
                                test_percent=test_percent,
                                val_percent=val_percent,
                                batch_alpha=hparams.batch_alpha,
                                batch_size=hparams.batch_size,
                                num_workers=hparams.num_workers)
    else:
        print('Unknown dataset. Exiting.')
        return

    # model
    model = MammoNet(backbone=hparams.model, learning_rate=hparams.learning_rate)

    # example for loading a checkpoint
    # ckpt = torch.load('/vol/biomedic3/bglocker/mammo/mammo-net/output-vindr/resnet18-b32-fixed-alpha-1.0-lr-0.0001/version_0/checkpoints/epoch=3-step=1600.ckpt')['state_dict']
    # for key in list(ckpt.keys()):
    #     ckpt[key.replace('model.', '')] = ckpt.pop(key)
    # model = MammoNet(learning_rate=hparams.learning_rate, checkpoint=ckpt)
    

    # Create output directory
    output_dir = os.path.join(hparams.output_root,hparams.output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('')
    print('=============================================================')
    print('TRAINING...')
    print('=============================================================')
    print('')

    # train
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        accelerator='auto',
        devices=hparams.num_devices,
        precision='16-mixed',
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(hparams.output_root, name=hparams.output_name),
        callbacks=[ModelCheckpoint(monitor='val_auc', mode='max'), TQDMProgressBar(refresh_rate=10)],
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

    model_modified = MammoNetEmbeddings(init=model)
    trainer.test(model=model_modified, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)
    save_embeddings(model=model_modified, output_fname=os.path.join(output_dir, 'embeddings.csv'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_alpha', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='embed')
    parser.add_argument('--csv_file', type=str, default='data/embed-non-negative.csv')
    parser.add_argument('--output_root', type=str, default='output')
    parser.add_argument('--output_name', type=str, default='debug')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)
