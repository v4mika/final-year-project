import pandas as pd
import os
import re

from pathlib import Path
from data_utils import *
from enum import Enum

class TestType(Enum):
    BALANCED = 'Balanced'
    PROPORTIONAL = 'Proportional'

# TODO: create vindr test set

def load_csv(image_dir):
    try:
        mydf = pd.read_csv(embed_csv_dir, low_memory=False)

    except FileNotFoundError:
        print(csv_error)

    mydf["shortimgpath"] = mydf["image_path"]
    mydf["image_path"] = mydf["image_path"].apply(lambda x: image_dir / str(x))
    mydf["manufacturer_domain"] = mydf.Manufacturer.apply(lambda x: domain_maps[x])
    mydf["tissueden"] = mydf.tissueden.apply(lambda x: tissue_maps[x])
    mydf["SimpleModelLabel"] = mydf.ManufacturerModelName.apply(
        lambda x: modelname_map[x]
    )
    mydf["ViewLabel"] = mydf.ViewPosition.apply(lambda x: 0 if x == "MLO" else 1)
    mydf["CviewLabel"] = mydf.FinalImageType.apply(lambda x: 0 if x == "2D" else 1)

    mydf = mydf.dropna(
        subset=[
            "age_at_study",
            "tissueden",
            "SimpleModelLabel",
            "ViewLabel",
            "image_path",
        ]
    )
    return mydf

def get_train_csv(use_counterfactuals=False, only_counterfactuals=False, save=False, save_path=None):
    image_dir = Path(embed_counterfactuals_dir)

    mydf = load_csv(image_dir)
    
    new_rows = []

    for _, row in mydf.iterrows():
        
        # Get the corresponding image subdirectory
        subdir = image_dir / Path(row["image_path"]).with_suffix("")
        if not subdir.exists():
            continue

        # Get all .png files in this directory
        png_files = list(subdir.glob("*.png"))
        if not png_files:
            continue
        
        for img_path in png_files:
            
            if use_counterfactuals:
                match = re.search(r'(?:trueclass|targetclass)_(\d)', img_path.name)
            elif only_counterfactuals:
                match = re.search(r'targetclass_(\d)', img_path.name)
            else:
                match = re.search(r'trueclass_(\d)', img_path.name)
                
            if not match:
                continue
                
            new_row = row.copy()
            new_row["image_path"] = str(img_path)
            new_row["tissueden"] = int(match.group(1))
            new_rows.append(new_row)

    expanded_df = pd.DataFrame(new_rows)
    
    print(expanded_df.tissueden.value_counts())
    print(f"Total number of images: {sum(expanded_df.tissueden.value_counts())}")

    if save:
        expanded_df.to_csv(Path(save_path), index=False)

    return expanded_df
    
class EmbedDensityTestCSVBuilder:
    def __init__(self, test_data_size=10000, seed=42):
        self.test_data_size = test_data_size
        self.seed = seed

    def build_csv(self, proportions, save=False, save_path=None):
        mydf = load_csv(image_dir=Path())

        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        print(mydf['tissueden'].value_counts())

        new_rows = []

        mydf = mydf.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        for _, row in mydf.iterrows():
            if row['FinalImageType'] != '2D':
                continue # skip if FinalImageType is not 2D

            density_class = row["tissueden"]
            if class_counts[density_class] >= proportions[density_class]:
                continue

            img_pth = Path(row["image_path"])
            study_id = img_pth.parts[-2]
            train_image_path = Path(embed_counterfactuals_dir) / Path(study_id)
            if os.path.isdir(train_image_path):
                continue

            real_img_path = Path(embed_data_dir) / row["shortimgpath"]
            if not real_img_path.exists():
                print(real_img_path)
                continue

            new_row = row.copy()
            new_rows.append(new_row)
            class_counts[density_class] += 1

            if sum(class_counts.values()) >= self.test_data_size:
                break

        test_df = pd.DataFrame(new_rows)

        if save:
            test_df.to_csv(save_path, index=False)
            print(f"Saved balanced test set with {len(test_df)} real images to {save_path}")

        print(test_df.tissueden.value_counts())
        return test_df

    def get_balanced_csv(self, save=False, save_path=None):
        class_target = self.test_data_size // 4
        proportions = {0: class_target, 1: class_target, 2: class_target, 3: class_target}
        print(f'Density class counts balanced: {proportions}')
        return self.build_csv(proportions, save, save_path)
        
    def get_proportional_csv(self, save=False, save_path=None):
        proportions = {k: round(v * self.test_data_size) for k, v in embed_density_proportions.items()}
        print(f'Density class counts proportional to EMBED dataset: {proportions}')
        return self.build_csv(proportions, save, save_path)
    
    def get_local_test_csv(self, type: TestType, save_path, save=False):
        class_target = self.test_data_size // 4
        proportions = ({0: class_target, 1: class_target, 2: class_target, 3: class_target}
                       if type == TestType.BALANCED 
                       else {k: round(v * self.test_data_size) for k, v in embed_density_proportions.items()})
        
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        image_dir = Path(local_test_data)

        mydf = load_csv(image_dir)
        
        new_rows = []
        i = 0
        for _, row in mydf.iterrows():

            if row['FinalImageType'] != '2D':
                continue # skip if FinalImageType is not 2D
            
            density_class = row["tissueden"]
            if class_counts[density_class] >= proportions[density_class]:
                continue

            # make sure not to mix studies between test/train data
            img_pth = Path(row["image_path"])
            study_id = img_pth.parts[-2]
            train_image_path = Path(embed_counterfactuals_dir) / Path(study_id)
            
            # Get the corresponding image subdirectory
            local_file = image_dir / Path(row["image_path"])
            if not os.path.exists(local_file):
                continue
            
            if os.path.isdir(train_image_path):
                continue
    
            i += 1
            new_row = row.copy()
            new_rows.append(new_row)
            class_counts[density_class] += 1

            if sum(class_counts.values()) >= self.test_data_size:
                break

        expanded_df = pd.DataFrame(new_rows)
        print(expanded_df.head)
        
        print(expanded_df.tissueden.value_counts())
        print(f"Total number of images: {sum(expanded_df.tissueden.value_counts())}")

        if save:
            expanded_df.to_csv(Path(save_path), index=False)

        return expanded_df
    
    def get_test_csv(self, type : TestType, image_dir, save_path, save=False):
        match type: 
            case TestType.BALANCED:
                return self.get_balanced_csv(save, save_path, image_dir)
            case TestType.PROPORTIONAL:
                return self.get_proportional_csv(save, save_path, image_dir)
        
if '__main__' == __name__:
    # # cf + real csv
    # get_train_csv(save=True, save_path='cf-and-real-embed-train.csv', use_counterfactuals=True)
    # # only real csv
    # get_train_csv(save=True, save_path='real-embed-train.csv', use_counterfactuals=False)
    # # only cf csv
    # get_train_csv(save=True, save_path='cf-embed-train.csv', only_counterfactuals=True)

    test_csv_builder = EmbedDensityTestCSVBuilder(test_data_size=5000)
    # # balanced test set of 5000 images
    # test_csv_builder.get_test_csv(type=TestType.BALANCED, save=True, save_path='balanced-5000-test.csv')
    # # proportional test set of 5000 images
    # test_csv_builder.get_test_csv(type=TestType.PROPORTIONAL, save=True, save_path='proportional-5000-test.csv')
    test_csv_builder.get_local_test_csv(type=TestType.BALANCED, save=True, save_path='local-balanced-5000-test.csv')
    test_csv_builder.get_local_test_csv(type=TestType.PROPORTIONAL, save=True, save_path='proportional-5000-test.csv')