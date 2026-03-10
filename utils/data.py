import os.path as osp

import PIL.Image as PImage
from dataloader.traindataset import TrainDataset

from dataloader.testdataset import TestDataset


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    valid_data_path: str, final_reso: int,
    tokenizer, null_text_ratio, original_image_ratio, args
):
    # build augmentations
    train_set = TrainDataset(hr_folder=args.hr_folder, lr_folder=args.lr_folder, crop_size=args.crop_size, lr_scale=args.lr_scale)
    val_set = TestDataset(valid_data_path, image_size=final_reso, tokenizer=tokenizer, resize_bak=True)
    num_classes = 2
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')    
    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
