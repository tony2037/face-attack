from torch.utils.data import Dataset
from glob import glob
import os, sys
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class DatasetAB(Dataset):
    
    def __init__(self, path_a, path_b, format_mode='jpg', samples=100, device=None):
        
        self.samples = samples
        self.device = None
        if device is not None:
            self.device = device
        self.transform = transforms.ToTensor()
        imgs_a = glob(os.path.join(path_a, '*.%s' % format_mode))
        imgs_b = glob(os.path.join(path_b, '*.%s' % format_mode))
        self.data_a = []
        self.data_b = []
        self.data = []

        for a in imgs_a:
            img = self.image_to_tensor(a)
            self.data_a.append(img)

        for b in imgs_b:
            img = self.image_to_tensor(b)
            self.data_b.append(img)

        for b in self.data_b:
            for a in self.data_a:
                self.data.append([a, b])
            if (len(self.data) >= self.samples):
                break

    def __getitem__(self, item):
        return self.data[item]
    
    def __len__(self):
        return len(self.data)

    def get_all(self):
        return np.array(self.data_a), np.array(self.data_b)

    def image_to_tensor(self, path):

        img = Image.open(path).convert('RGB')
        return self.transform(img).to(device=self.device)
