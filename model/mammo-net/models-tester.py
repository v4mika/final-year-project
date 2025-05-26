import os
from pathlib import Path
import re
import torch

from data_utils import embed_data_dir
from pytorch_lightning.callbacks import ModelCheckpoint
from mammo_net_density import *

image_size = (224, 168) 

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

def test_existing_models(ckpts_path=Path('/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/refined_sweep/'), seed=42):
    # torch.set_float32_matmul_precision('medium')
    torch.set_float32_matmul_precision('high')
                                       
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(seed, workers=True)

    for ckpt_path in ckpts_path.iterdir():

        if ckpt_path.is_dir():
            print(f'testing model in: {ckpt_path}')

        hparams = extract_hparams_from_foldername(os.path.basename(ckpt_path))

        output_dir = os.path.join(ckpt_path,'test_existing_models')

        # find the best checkpoint
        ckpt_files = list(Path(ckpt_path/'version_0'/'checkpoints').glob('*.ckpt'))
        if not ckpt_files:
            print(f"no checkpoint found for {ckpt_path}")
            continue
        ckpt_path = str(ckpt_files[0])  
        ckpt = torch.load(ckpt_path)['state_dict']
        for key in list(ckpt.keys()):
            ckpt[key.replace('model.', '')] = ckpt.pop(key)
        model = MammoNet(backbone=hparams['model'], learning_rate=hparams['learning_rate'], checkpoint=ckpt)

        data = EMBEDMammoDataModule(data_dir=embed_data_dir,
                                image_size=image_size,
                                test_percent=test_percent,
                                val_percent=val_percent,
                                batch_alpha=hparams['batch_alpha'],
                                batch_size=hparams['batch_size'],
                                num_workers=hparams['num_workers'], 
                                use_counterfactuals=hparams['cf'])
        
        trainer = pl.Trainer(
            max_epochs=hparams['epochs'],
            accelerator='auto',
            devices=hparams['num_devices'],
            precision='16-mixed',
            num_sanity_val_steps=0,
            logger=TensorBoardLogger(output_dir, name='debug'),
            callbacks=[
                ModelCheckpoint(monitor='val_auc', mode='max'), 
                TQDMProgressBar(refresh_rate=10) 
                ],
        )

        print('TESTING...')
        trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)
        restored_model = trainer.lightning_module
        save_predictions(model=restored_model, output_fname=os.path.join(output_dir, 'predictions.csv'))


        print('EMBEDDINGS...')
        model_modified = MammoNetEmbeddings(init=model)
        trainer.test(model=model_modified, datamodule=data, ckpt_path=ckpt_path)
        restored_model = trainer.lightning_module
        save_embeddings(model=restored_model, output_fname=os.path.join(output_dir, 'embeddings.csv'))

if __name__ == '__main__':
    test_existing_models()