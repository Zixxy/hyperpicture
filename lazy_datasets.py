from glob import glob
from tqdm import tqdm
from PIL import Image
from crops_utils import *
from utils import get_interp_function

import tensorflow as tf
import scipy.misc
import numpy as np

class LazyImagesGenerator:
    def __next_img_crops(self, img_path):
        x = []
        y = []
        img = np.array(Image.open(img_path))
        if self.hparams.crop_images:
            w, h, c = img.shape
            c_x = self.hparams.crop_x_size
            c_y = self.hparams.crop_y_size
            n_w = w // c_x
            n_h = h // c_y
            i = 0
            while(i < n_w):
                j = 0
                while(j < n_h):
                    crop = img[i * c_x : (i+1) * c_x, j * c_y : (j+1) * c_y, :]
                    x.append(crop)
                    y.append(np.array(crop))
                    j += 1
                i += 1

            def extend_lists(imgs):
                x.extend(np.array(imgs))
                y.extend(np.array(imgs))

            if w % c_x != 0:
                begins = [i * c_x for i in range(w // c_x)]
                begins.append((w - c_x))
                crops = [img[b : b + c_x, h - c_y : h, :] for b in begins]
                extend_lists(crops)
            if h % c_y != 0:
                begins = [i * c_y for i in range(h // c_y)]
                begins.append((h - c_y))
                crops = [img[w - c_x : w, b : b + c_y, :] for b in begins]
                extend_lists(crops)
        else:
            x.append(np.array(img))
            y.append(np.array(img))
        return x, y
        
    def __data_augment(self, x, y):
        def horizontal_flip(b):
            a = np.flip(b, axis=2)
            return np.concatenate([a,b])
        x = horizontal_flip(x)
        y = horizontal_flip(y)
        return x,y

    def __normalize_colors(self, x, y):
        return x / 255, y / 255
    
    def __multiple_scales(self):
        if not self.hparams.random_scales:
            return
        print('random scales')
        assert self.hparams.in_img_width == self.hparams.in_img_height 
        find_divisors = self.hparams.in_img_width 
        max_scale = int(self.hparams.scale) 
        assert max_scale == self.hparams.scale
        i = 0
        while(2**i < find_divisors):
            i+=1
            
        if 2**i > find_divisors:
            raise Exception("dimension is not power of 2!")
        max_i = i
        self.scale_values = [j + 1/2**i for j in range(max_scale) for i in range(max_i)]
        self.scale_values = [2,3,4] #tmp
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.__multiple_scales()
        
    def generator(self):
        prev_x, prev_y = [], []
        batch_size = self.hparams.batch_size
        scale_down_function = get_interp_function(self.hparams.scale_down_func)
        imgs = glob(self.hparams.train_dataset_path + '/*')
        imgs = sorted(imgs)
        imgs = np.array(imgs)
        imgs_downscaled = None
        if self.hparams.train_dataset_path_downscaled != '':
            imgs_downscaled = glob(self.hparams.train_dataset_path_downscaled + '/*')
            imgs_downscaled = sorted(imgs_downscaled)
            if len(imgs_downscaled) != len(imgs):
                raise Exception("Incompatibile train downscaled with HR files.")
            imgs_downscaled = np.array(imgs_downscaled)

        while True:
            if(len(imgs) == 0):
                raise Exception('empty images set!')
            p = np.random.permutation(len(imgs))
            imgs = imgs[p]
            if imgs_downscaled is not None:
                imgs_downscaled = imgs_downscaled[p]
                
            for img in range(len(imgs)):
                scale_in = self.hparams.scale_in
                if self.hparams.random_scales:
                    next_scale = np.random.randint(len(self.scale_values))
                    scale = self.scale_values[next_scale]
                else:
                    scale = self.hparams.scale
                
                if imgs_downscaled is None:
                    x, y = get_corresponding_crops(imgs[img], self.hparams.in_img_width, self.hparams.in_img_height, scale_in, scale, scale_down_function)
                else:
                    x, y = get_corresponding_crops(imgs[img], self.hparams.in_img_width, self.hparams.in_img_height, scale_in, scale, scale_down_function, imgs_downscaled[img])

                x, y = np.array(x), np.array(y)
                if self.hparams.data_augment:
                    x, y = self.__data_augment(x, y)
                x, y = self.__normalize_colors(x, y)
                x, y = list(x), list(y)
                
                for i in range(len(x) // batch_size):
                    yield (x[i*batch_size: (i+1) * batch_size], y[i*batch_size: (i+1) * batch_size], scale)
                
                int_val = len(x) // batch_size
                int_val *= batch_size
                if len(x[int_val:]) > 0:
                    yield (x[int_val:], y[int_val:], scale)
                
class LazyDataset:
    def __prepare_datasets(self):
        print('Initializing dataset...')
        hparams = self.hparams
        self.train_generator = LazyImagesGenerator(hparams)
        

    def __init__(self, hparams):
        self.hparams = hparams
        self.__prepare_datasets()
                
    def train_data_generator(self):
        return self.train_generator.generator()
    

