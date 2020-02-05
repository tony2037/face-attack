from glob import glob
import os, sys
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

class Loader():
    
    def __init__(self, path_a, path_b, format_mode='jpg', samples=100):
        
        self.samples = samples
        imgs_a = glob(os.path.join(path_a, '*.%s' % format_mode))
        imgs_b = glob(os.path.join(path_b, '*.%s' % format_mode))
        self.data_a = []
        self.data_b = []
        self.data = []

        for a in imgs_a:
            img = load_img(a)
            img = img_to_array(img)
            self.data_a.append(img)

        for b in imgs_b:
            img = load_img(b)
            img = img_to_array(img)
            self.data_b.append(img)

        for b in self.data_b:
            for a in self.data_a:
                self.data.append([a, b])
            if (len(self.data) >= self.samples):
                break

    def get_all(self):
        return np.array(self.data_a), np.array(self.data_b)
