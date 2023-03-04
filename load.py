import pandas as pd
import numpy as np
import os
import shutil
import torchvision
from torchvision import transforms
import torch.utils.data as data


def prepare_images():
    data_dir = f"{os.getcwd()}/data/HAM10000_images"
    dest_dir = f"{os.getcwd()}/data/organized_images"

    skin_df = pd.read_csv("data/HAM10000_metadata.csv")
    labels = skin_df["dx"].unique().tolist()
    label_images = []

    for label in labels:
        target_dir = f"{dest_dir}/{label}/"
        os.makedirs(target_dir, exist_ok=True)
        sample = skin_df[skin_df["dx"] == label]["image_id"]
        label_images.extend(sample)
        for item in label_images:
            shutil.copyfile(f"{data_dir}/{item}.jpg", f"{target_dir}/{item}.jpg")
        label_images = []


def load_images():
    train_dir = f"{os.getcwd()}/data/organized_images"
    transform_img = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    train_data = torchvision.datasets.ImageFolder(
        root=train_dir, transform=transform_img
    )
    train_data_loader = data.DataLoader(train_data, batch_size=len(train_data))

    labels = np.array(train_data.targets)
    (unique, counts) = np.unique(labels, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    # print(f"Number of train samples: {len(train_data)}")
    # print(f"Detected classes: {train_data.class_to_idx}")
    # print(f"Freq: {frequencies}")


if __name__ == "__main__":
    prepare_images()
    load_images()
