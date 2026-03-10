import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from .dataset import Dataset_LPM
from torchvision.transforms import InterpolationMode, transforms
import pickle


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125, use_lpm=False,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    num_classes = 1000

    if use_lpm:
        file_path = '/home/lpm/image_attrs/balanced/1000w-prior100w-k4000-per1200.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        num_classes = len(data)
        train_set = Dataset_LPM(root=data, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug, num_classes=num_classes, img_start=0, img_end=50)
        val_set = Dataset_LPM(root=data, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug, num_classes=num_classes, img_start=1, img_end=50)
    else:
        # build dataset
        train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
        val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
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
