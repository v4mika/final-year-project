import pandas as pd
import re

from pathlib import Path
from data_utils import *
from enum import Enum

class TestType(Enum):
    BALANCED = 'Balanced'
    PROPORTIONAL = 'Proportional'

class EmbedCSVGenerator:
    def __init__(self, image_dir=embed_counterfactuals_dir, use_counterfactuals=False, csv_path='embed-generated.csv', train_data_size=None):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.use_counterfactuals = use_counterfactuals
        self.train_data_size = train_data_size

    def load_csv(self):
        image_dir = Path(self.image_dir)
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

    def get_train_csv(self, save=False):
        image_dir = Path(self.image_dir)

        mydf = self.load_csv()
        
        new_rows = []
        data_processed = 0

        for _, row in mydf.iterrows():
            if self.train_data_size and data_processed >= self.train_data_size:
                break
            
            # Get the corresponding image subdirectory
            subdir = image_dir / Path(row["image_path"]).with_suffix("")
            if not subdir.exists():
                continue

            # Get all .png files in this directory
            png_files = list(subdir.glob("*.png"))
            if not png_files:
                continue
            
            for img_path in png_files:
                
                if self.use_counterfactuals:
                    match = re.search(r'(?:trueclass|targetclass)_(\d)', img_path.name)
                else:
                    match = re.search(r'trueclass_(\d)', img_path.name)
                    
                if not match:
                    continue
                    
                new_row = row.copy()
                new_row["image_path"] = str(img_path)
                new_row["tissueden"] = int(match.group(1))
                new_rows.append(new_row)
                data_processed += 1

        expanded_df = pd.DataFrame(new_rows)
        
        print(expanded_df.tissueden.value_counts())
        print(f"Total number of images: {sum(expanded_df.tissueden.value_counts())}")

        if save:
            expanded_df.to_csv(self.csv_path, index=False)

        return expanded_df
    
class EmbedDensityTestCSVBuilder:
    def __init__(self, test_data_size=1000, seed=42):
        self.test_data_size = test_data_size
        self.seed = seed
        self.generator = EmbedCSVGenerator(image_dir=embed_data_dir, csv_path='embed-density-test.csv')

    def build_csv(self, proportions, save=False, output_csv=None):
        mydf = self.generator.load_csv()

        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        new_rows = []

        mydf = mydf.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        for _, row in mydf.iterrows():
            if row['FinalImageType'] != '2D':
                continue # skip if FinalImageType is not 2D

            density_class = row["tissueden"]
            if class_counts[density_class] >= proportions[density_class]:
                continue

            cf_subdir = self.generator.image_dir / Path(row["shortimgpath"]).with_suffix("")
            if cf_subdir.exists():
                continue  # skip if counterfactuals for that image exist

            real_img_path = Path(self.generator.image_dir) / row["shortimgpath"]
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
            output_csv = f'balanced-{self.generator.csv_path}'
            test_df.to_csv(output_csv, index=False)
            print(f"Saved balanced test set with {len(test_df)} real images to {output_csv}")

        print(test_df.tissueden.value_counts())
        return test_df

    def get_balanced_csv(self, save=False):
        class_target = self.test_data_size // 4
        proportions = {0: class_target, 1: class_target, 2: class_target, 3: class_target}
        print(f'Density class counts balanced: {proportions}')
        output_csv = None if not save else f'balanced-{self.generator.csv_path}'
        return self.build_csv(proportions, save, output_csv)
        
    def get_proportional_csv(self, save=False):
        output_csv = None if not save else f'proportional-{self.generator.csv_path}'
        proportions = {k: (v * self.test_data_size) for k, v in embed_density_proportions.items()}
        print(f'Density class counts proportional to EMBED dataset: {proportions}')
        return self.build_csv(proportions, save, output_csv)
    
    def get_test_csv(self, type : TestType, save=False):
        match type: 
            case TestType.BALANCED:
                return self.get_balanced_csv(save)
            case TestType.PROPORTIONAL:
                return self.get_proportional_csv(save)

# TODO: delete this main function
if '__main__' == __name__:
    gen = EmbedDensityTestCSVBuilder()
    df = gen.get_proportional_csv()
    print(df['FinalImageType'])

# TODO delete this list
# to test csv logic want to ensure that FinalImageType == 2D for all images in test set 
# need to preprocss the images changing them from 


# tsfm = CenterCrop((224,168))
# gt_img = tsfm((batch["x"][img_idx][0] + 1) / 2)  # Rescale from [-1,1] to [0,1]
