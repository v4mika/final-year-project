import os
import copy

from argparse import Namespace
from mammo_net_density import main

epochs = [5]
batches_lrs = [(64, 1e-4), (128, 3e-4), (256, 5e-4)]
models = ['resnet18', 'resnet32', 'resnet50']
train_csv_files = [
    ('/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/csv/train/cf-and-real-embed-train.csv', 'cf_and_real'),
    ('/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/csv/train/cf-embed-train.csv', 'cf_only'),
    ('/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/csv/train/real-embed-train.csv', 'real_only')
]

def run_hyperparameter_sweep():

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
        output_root='final_models',
        output_name='debug',
        seed=42,
        counterfactuals=False,
        max_data_size=None
    )
    
    for num_epoch in epochs:
        for batch, lr in batches_lrs:
            for model in models:
                for train_csv_file, name in train_csv_files:
                    output_name = f"{name}_{model}_bs{batch}_lr{lr}_epochs{num_epoch}"
                    output_path = os.path.join(base_args.output_root, output_name)

                    if os.path.exists(output_path):
                        print(f"skipping existing run: {output_name}")
                        continue

                    args = copy.deepcopy(base_args)
                    args.train_csv_file = train_csv_file
                    args.test_csv_file = '/vol/biomedic3/bglocker/ugproj/vg521/model/mammo-net/local-balanced-3397-test.csv'
                    args.epochs = num_epoch
                    args.batch_size = batch
                    args.learning_rate = lr

                    args.output_name = output_name

                    print(f"Running with: {args.output_name}")
                    main(args)
                    
if __name__ == '__main__':
    run_hyperparameter_sweep()