import os
import glob
import torch
from PIL import Image

from torchvision import transforms
from torch.utils import data as data
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
    
import os
import glob
import torch
from PIL import Image

from torchvision import transforms
from torch.utils import data as data
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
       
    
class TrainDataset(data.Dataset):
    def __init__(self, hr_folder="FLIR-IISR/HR", lr_folder="FLIR-IISR/LR", crop_size=512, lr_scale=4,):
        super(TrainDataset, self).__init__()
        
        self.crop_size = crop_size
        exts = ('*.bmp', '*.png', '*.jpg', '*.jpeg')
        self.hr_paths = []
        self.lr_paths = []
        for ext in exts:
            self.hr_paths.extend(glob.glob(os.path.join(hr_folder, ext)))
            self.lr_paths.extend(glob.glob(os.path.join(lr_folder, ext)))
        self.hr_paths = sorted(self.hr_paths)
        self.lr_paths = sorted(self.lr_paths)

        assert len(self.hr_paths) == len(self.lr_paths), "The number of images in the HR and LR folders must be the same"

        self.labels = torch.cat([
            torch.zeros(len(self.hr_paths)),
            torch.ones(len(self.lr_paths))
        ], dim=0).tolist()
        self.img_paths = self.hr_paths + self.lr_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label_B = int(self.labels[index])

        img_hr = Image.open(img_path).convert('RGB')
        img_path_lr = img_path.replace("HR", "LR")
        img_lr = Image.open(img_path_lr).convert('RGB')

        i, j, h, w = transforms.RandomCrop.get_params(img_hr, output_size=(self.crop_size, self.crop_size))
        crop_hr = TF.crop(img_hr, i, j, self.crop_size, self.crop_size)
        crop_lr = TF.crop(img_lr, i, j, self.crop_size, self.crop_size)

        HR = TF.to_tensor(crop_hr)
        LR_small = TF.resize(crop_lr, [self.crop_size // 4, self.crop_size // 4], interpolation=Image.BICUBIC)
        LR_upsampled = TF.resize(LR_small, [self.crop_size, self.crop_size], interpolation=Image.BICUBIC)
        LR = TF.to_tensor(LR_upsampled)

        example = {
            "HR": HR * 2.0 - 1.0,
            "LR": LR * 2.0 - 1.0,
            "label_B": label_B,
            "img_path": img_path
        }
        return example
