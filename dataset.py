import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder


train_transform = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(),
            A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3
            ),
            A.FancyPCA(alpha=0.1)
        ], p=0.75),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
                    scale_limit=[-0.1, 0.1],
                    rotate_limit=15,
                    shift_limit=0.06,
                    p=0.7,
                    border_mode=0),
        ToTensorV2()
    ])
val_transform = A.Compose([
    A.Resize(224, 224),
    ToTensorV2()
])


class Celeba(Dataset):
    def __init__(self, image_dir, csv_path=None, test_mode=False, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.image_to_label = {}
        self.test_mode = test_mode
        self.label_encoder = LabelEncoder()

        # remove path part to jpg from csv column ['image']
        csv = pd.read_csv(csv_path)
        filename = os.path.basename(csv_path).split('.')[0]
        csv['image'] = csv['image'].str.replace(f'celeba/{filename}/', '')

        # create dict Image: Label
        for _, row in csv.iterrows():
            image_name = row['image']
            label = row['spoof_label']
            self.image_to_label[image_name] = label
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.test_mode:
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image
        else:
            label = self.image_to_label[self.images[index]]
            label = torch.tensor(label, dtype=torch.float32)
            label = label.unsqueeze(0)
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image.numpy().astype(np.float32)/255.0, label

