from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torchvision.transforms import Resize, CenterCrop
import os

from torch.utils.data import DataLoader

if Path("/data2/mb121/EMBED").exists():
    EMBED_ROOT = "/data2/mb121/EMBED"
if Path("/data/EMBED").exists():
    EMBED_ROOT = "/data/EMBED"
else:
    EMBED_ROOT = "/vol/biomedic3/data/EMBED"
    
VINDR_MAMMO_DIR = Path("/vol/biomedic3/data/VinDR-Mammo")

domain_maps = {
    "HOLOGIC, Inc.": 0,
    "GE MEDICAL SYSTEMS": 1,
    "FUJIFILM Corporation": 2,
    "GE HEALTHCARE": 3,
    "Lorad, A Hologic Company": 4,
}

tissue_maps = {"A": 0, "B": 1, "C": 2, "D": 3}
modelname_map = {
    "Selenia Dimensions": 0,
    "Senographe Essential VERSION ADS_53.40": 5,
    "Senographe Essential VERSION ADS_54.10": 5,
    "Senograph 2000D ADS_17.4.5": 2,
    "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3,
    "Clearview CSm": 1,
    "Senographe Pristina": 4,
}


def preprocess_breast(image_path, target_size):
    """
    Loads the image performs basic background removal around the breast.
    Works for text but not for objects in contact with the breast (as it keeps the
    largest non background connected component.)
    """
    image = cv2.imread(str(image_path))

    if image is None:
        # sometimes bug in reading images with cv2
        from skimage.util import img_as_ubyte

        image = io.imread(image_path)
        gray = img_as_ubyte(image.astype(np.uint16))
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

    # Connected components with stats.
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=4
    )

    # Find the largest non background component.
    # Note: range() starts from 1 since 0 is the background label.
    max_label, max_area = max(
        [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
        key=lambda x: x[1],
    )
    mask = output == max_label
    area = mask.astype(float).mean()
    img = torch.tensor((gray * mask) / 255.0).unsqueeze(0).float()
    img = Resize(target_size, antialias=True)(img)
        
    return img, area


def get_embed_csv():
    image_dir = EMBED_ROOT / Path("images/png/1024x768")
    try:
        mydf = pd.read_csv("/vol/biomedic3/mb121/tech-demo/code_for_demo/joined_simple.csv")
    except FileNotFoundError:
        print(
            """
            For running EMBED code you need to first generate the csv
            file used for this study in csv_generation_code/generate_embed_csv.ipynb
            """
        )

    mydf["shortimgpath"] = mydf["image_path"]
    mydf["image_path"] = mydf["image_path"].apply(lambda x: image_dir / str(x))

    mydf["manufacturer_domain"] = mydf.Manufacturer.apply(lambda x: domain_maps[x])

    # convert tissueden to trainable label
    mydf["tissueden"] = mydf.tissueden.apply(lambda x: tissue_maps[x])

    mydf["SimpleModelLabel"] = mydf.ManufacturerModelName.apply(
        lambda x: modelname_map[x]
    )
    print(mydf.SimpleModelLabel.value_counts())
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
    # artefact_dataset = pd.read_csv('/vol/biomedic3/mb121/tech-demo/data_handling/predicted_all_embed.csv')
    # artefact_dataset['image_path'] = artefact_dataset['image_path'].apply(lambda x: x.replace('/data/EMBED/images/png/1024x768/', ''))
    # reject_images = artefact_dataset.loc[(artefact_dataset['breast implant'] == 1) | (artefact_dataset['compression']== 1), 'image_path'].values
    # mydf = mydf.loc[~mydf['shortimgpath'].isin(reject_images)]
    return mydf


class EmbedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: torch.nn.Module,
        parents,
        target_size=[224,168],
        label="tissueden",
    ) -> None:
        self.imgs_paths = df.image_path.values
        self.shortpaths = df.shortimgpath.values
        self.labels = df[label].values
        self.target_size = target_size
        self.transform = transform
        self.views = df.ViewLabel.values
        self.scanner = df.SimpleModelLabel.values
        self.cview = df.FinalImageType.apply(lambda x: 0 if x == "2D" else 1).values
        self.age = df.age_at_study.values
        self.parents = parents
        self.densities = df.tissueden.values
        self.areas = np.zeros_like(self.densities) * np.nan

    def __getitem__(self, index) -> Any:
        img, area = preprocess_breast(self.imgs_paths[index], self.target_size)
        sample = {}
        age = self.age[index]
        sample['area'] = float(area)
        sample["cview"] = self.cview[index]
        sample["shortpath"] = str(self.shortpaths[index])
        sample["real_age"] = age
        sample["view"] = self.views[index]
        sample["density"] = torch.nn.functional.one_hot(
            torch.tensor(self.densities[index]).long(), num_classes=4
        ).detach()
        sample["y"] = self.labels[index]
        sample["scanner_int"] = self.scanner[index]
        sample["scanner"] = torch.nn.functional.one_hot(
                torch.tensor(self.scanner[index]).long(), num_classes=4
            ).detach()
        # Only used for causal models
        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()
        img = self.transform(img)
        img = CenterCrop((224, 168))(img)
        sample["x"] = img.float()
        return sample

    def __len__(self):
        return self.labels.shape[0]


class EmbedDataModule(LightningDataModule):
    def __init__(self, parents) -> None:
        self.parents = parents
        full_df = get_embed_csv()
        # Keep senograph essential a hold-out set
        full_df = full_df[full_df["SimpleModelLabel"] != 5]
        # remove pristina 0.2% train set
        self.full_df = full_df[full_df["SimpleModelLabel"] != 4]

        split_csv_dict = self.get_all_csv_from_config(
            orig_df=self.full_df,
        )

        inference_tsfm = CenterCrop((224,224))
        self.dataset_train = EmbedDataset(
            df=split_csv_dict["train"],
            transform=inference_tsfm,
            label='tissueden',
            parents=self.parents,
        )

        self.dataset_val = EmbedDataset(
            df=split_csv_dict["val"],
            transform=inference_tsfm,
            label='tissueden',
            parents=self.parents,
        )

        self.dataset_test = EmbedDataset(
            df=split_csv_dict["test"],
            transform=inference_tsfm,
            label='tissueden',
            parents=self.parents,
        )

    @property
    def num_classes(self) -> int:
        return 4
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            12,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            12,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            12,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
        )
    
    @property
    def dataset_name(self) -> str:
        return "EMBED"

    def get_all_csv_from_config(
        self, orig_df,
    ):
        df = orig_df.copy()
        df = df.loc[df.FinalImageType == "2D"]

        train_id, val_id = train_test_split(
            df.empi_anon.unique(), test_size=0.15, random_state=33
        )

        # 15% of patients - 600 for test
        test_id = val_id[600:]

        # 600 patients for val
        val_id = val_id[:600]

        print("train n images" + str(len(df.loc[df.empi_anon.isin(train_id)])))
        print("test" + str(len(df.loc[df.empi_anon.isin(test_id)])))
        print("val" + str(len(df.loc[df.empi_anon.isin(val_id)])))
        return {
            "train": df.loc[df.empi_anon.isin(train_id)],
            "val": df.loc[df.empi_anon.isin(val_id)],
            "test": df.loc[df.empi_anon.isin(test_id)],
        }
        
def prepare_vindr_dataset():
    print('prepare vindr dataset')
    df = pd.read_csv(VINDR_MAMMO_DIR / "breast-level_annotations.csv")
    meta = pd.read_csv(VINDR_MAMMO_DIR / "metadata.csv")
    df["img_path"] = df[["study_id", "image_id"]].apply(
        lambda x: VINDR_MAMMO_DIR / "pngs" / x[0] / f"{x[1]}.png", axis=1
    )
    df["ViewLabel"] = df.view_position.apply(lambda x: 0 if x == "MLO" else 1)
    tissue_maps = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
    df["tissueden"] = df.breast_density.apply(lambda x: tissue_maps[x])
    df = pd.merge(df, meta, left_on="image_id", right_on="SOP Instance UID")
    df["Scanner"] = df["Manufacturer's Model Name"]
    df["Manufacturer"] = df["Manufacturer"]
    df.dropna(subset="tissueden", inplace=True)
    return df

class VinDRMammoDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: torch.nn.Module,
        target_size=[224,168],
        test_set=True,
    ) -> None:
        self.imgs_paths = df.img_path.values
        print(f"len df {self.imgs_paths.shape[0]}")
        self.labels = df["density_label"].values.astype(int)
        
        print(df["density_label"].value_counts())
        print(df["density_label"].value_counts(normalize=True))
        self.transform = transform
        self.target_size = target_size
        self.views = df.ViewLabel.values
        self.scanner = df.Scanner.values
        self.manufacturer = df.Manufacturer.values
        self.densities = df.tissueden.values
        self.laterality = df.laterality.values
        self.study_ids = df.study_id.values
        self.image_ids = df.image_id.values
        self.test_set = test_set

    def __getitem__(self, index) -> Any:
        img, area = preprocess_breast(self.imgs_paths[index], self.target_size)

        if self.laterality[index] == 'R':
            img = torch.flip(img, dims=[2])

        # Apply transform
        img = self.transform(img).float()

        # If test set → center crop and rescale from [-1,1] to [0,1]
        if self.test_set:
            tsfm = CenterCrop((224, 168))
            img = tsfm(img)

        # Repeat channels to get 3 channels
        img = img.repeat(3, 1, 1)

        # Return dict matching MammoDataset style
        return {
            'image': img,
            'label': self.labels[index],
            'study_id': self.study_ids[index],
            'image_id': self.image_ids[index],
        }

    def __len__(self):
        return self.labels.shape[0]

    def get_labels(self):
        return self.labels

# class VinDrDataModule(LightningDataModule):
#     def __init__(self) -> None:
#         full_df = prepare_vindr_dataset()
#         inference_tsfm = CenterCrop((224,224))
        
#         # only using for testing purposes
#         self.dataset_test = VinDRMammoDataset(
#             df=full_df,
#             transform=inference_tsfm,
#         )
        
#     @property
#     def dataset_name(self):
#         return "vindr"

#     @property
#     def num_classes(self):
#         return 4
    
#     def test_dataloader(self):
#         return DataLoader(
#             self.dataset_test,
#             12,
#             shuffle=True,
#             num_workers=12,
#             pin_memory=True
#         )