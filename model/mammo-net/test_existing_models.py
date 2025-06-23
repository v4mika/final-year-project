import csv
import os
from pathlib import Path
import re
import torch

from data_utils import embed_data_dir
from pytorch_lightning.callbacks import ModelCheckpoint
from mammo_net_density import *

test_csv_path = '/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/csv/test/local-balanced-3397-test.csv'
output_csv_name = 'same_class_balanced'
ckpts_path = '/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/same_class_models'

train_csv_files = [
    ('/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/csv/train/cf-and-real-embed-train.csv', 'cf_and_real'),
    ('/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/csv/train/cf-embed-train.csv', 'cf_only'),
    ('/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/csv/train/real-embed-train.csv', 'real_only'),
    ('/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/csv/train/same-class-cf-train.csv', 'same-class-cf')
]

train_csv_dict = {key: value for (value, key) in train_csv_files}

def parse_and_sort_tracker(csv_path, sort_by='test_auc', descending=True):
    results = []

    # Read the tracker CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Convert numeric fields
                row['batch_size'] = int(row['batch_size'])
                row['learning_rate'] = float(row['learning_rate'])
                row['epochs'] = int(row['epochs'])
                row['test_auc'] = float(row['test_auc'])
                row['test_loss'] = float(row['test_loss'])
            except ValueError:
                # Skip rows with invalid values
                continue
            results.append(row)

    # Sort
    results.sort(key=lambda x: x[sort_by], reverse=descending)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def extract_hparams_from_foldername(filename):
    filename = filename.strip()
    match = re.search(r'([^/]+)_bs(\d+)_lr([\d.]+)_epochs(\d+)', filename)
    if not match:
        print('checkpoint folder name does not match expected pattern')
    if match:
        name_model, batch_size, learning_rate, epochs = match.groups()
        
        parts = name_model.rsplit('_', 1)
        name, model = parts

        batch_size = int(batch_size)
        learning_rate = float(learning_rate)
        epochs = int(epochs)
        return {
            'name': name,
            'train_csv': train_csv_dict[name],
            'model': model,
            'batch_size': int(batch_size),
            'learning_rate': float(learning_rate),
            'epochs': int(epochs),
            'seed': 42,
            'num_devices': 1,
            'num_workers': 6,
            'batch_alpha': 1.0
        }

if __name__ == '__main__':
    results = []

    test_name = 'balanced-embed'

    completed_folders = set()
    with open('tracker.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed_folders.add(row['folder'])

    ckpts_path = Path(ckpts_path)
    # torch.set_float32_matmul_precision('medium')
    torch.set_float32_matmul_precision('high')
                                        
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(seed=42, workers=True)

    for ckpt_path in ckpts_path.iterdir():

        hparams = extract_hparams_from_foldername(os.path.basename(ckpt_path))
        
        if ckpt_path.is_dir():
            print(f'testing model in: {ckpt_path}')

        folder = ckpt_path.parts[-1]

        if folder in completed_folders:
            continue

        output_dir = os.path.join(ckpt_path,'test_existing_model_test')
        predictions_path = ckpt_path

        # find the best checkpoint
        ckpt_files = list(Path(ckpt_path/'version_0'/'checkpoints').glob('*.ckpt'))
        if not ckpt_files:
            print(f"no checkpoint found for {ckpt_path}")
            continue
        ckpt_path = str(ckpt_files[0])  
        ckpt = torch.load(ckpt_path)['state_dict']
        
        ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
        model = MammoNet(backbone=hparams['model'], learning_rate=hparams['learning_rate'], checkpoint=ckpt)
        
        data = EMBEDMammoDataModule(image_size=(224, 168),
                                        val_percent=0.2,
                                        batch_alpha=hparams['batch_alpha'],
                                        batch_size=hparams['batch_size'],
                                        train_csv=hparams['train_csv'],
                                        test_csv=test_csv_path,
                                        num_workers=hparams['num_workers']
                                        )
        
        trainer = pl.Trainer(
            max_epochs=hparams['epochs'],
            accelerator='auto',
            devices=hparams['num_devices'],
            precision='16-mixed',
            num_sanity_val_steps=0,
            callbacks=[
                ModelCheckpoint(monitor='val_auc', mode='max'), 
                TQDMProgressBar(refresh_rate=10) 
                ],
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)

        print('TESTING...')
        test_result = trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)[0]
        save_predictions(model=model, output_fname=os.path.join(predictions_path, 'predictions_balanced.csv'))
        test_loss, test_auc = test_result.get('test_loss', 'NA'), test_result.get('test_auc', 'NA'),

        result = {
            'name': hparams['name'],
            'model': hparams['model'],
            'batch_size': hparams['batch_size'],
            'learning_rate': hparams['learning_rate'],
            'epochs': hparams['epochs'],
            'test_auc': test_auc,
            'test_loss': test_loss,
            'folder': folder
        }

        results.append(result)

        with open('tracker.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
    # parse_and_sort_tracker(csv_path='/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/tracker.csv')
