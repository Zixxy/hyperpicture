from glob import glob
from tqdm import tqdm
from PIL import Image
from skimage.color import gray2rgb

import tensorflow as tf
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

def __get_two_img_scale(img, down_scale_in, down_scale_2, resize_f):
    h, w, c = img.shape
    if down_scale_in == 1 and (down_scale_2 == 1):
        return np.array(img), img
    x = resize_f(img, (int(w // down_scale_in), int(h // down_scale_in))) 
    if (down_scale_2 == 1):
        return x, img
    y = resize_f(img, (int(w // down_scale_2), int(h // down_scale_2))) 
    return x, y


def __get_img_frame(img_in, x, y, img_out, scale):
    w,h,c = img_in.shape
    w_o, h_o, c_o = img_out.shape
    xs, ys = [], []
    x_scale = int(x*scale)
    y_scale = int(y*scale)
    if w % x != 0:
        begins = [i * x for i in range(w // x)]
        crops = [img_in[b : b + x, h - y : h, :] for b in begins]
        crops.append(img_in[w - x : w, h - y : h, :])
        xs.extend(crops)
        crops = [img_out[int(b*scale) : int((b + x)*scale), h_o - y_scale : h_o, :] for b in begins]
        crops.append(img_out[w_o - x_scale : w_o, h_o - y_scale : h_o, :])
        ys.extend(crops)
    if h % y != 0:
        begins = [i * y for i in range(h // y)]
        crops = [img_in[w - x : w, b : b + y, :] for b in begins]
        xs.extend(crops)
        crops = [img_out[w_o - x_scale : w_o, int(b*scale) : int((b + y)*scale), :] for b in begins]
        ys.extend(crops)
    return xs,ys

def __next_img_crops(img, img_out, crop, img_x, img_y, scale):
    x = []
    y = []
    if True: #for handling cifar other way :F
        w, h, c = img.shape
        w_o, h_o, c_o = img_out.shape
        c_x, c_y = img_x, img_y
        c_o_x, c_o_y = int(img_x*scale), int(img_y*scale)
        n_w = min(w // c_x, w_o // c_o_x)
        n_h = min(h // c_y, h_o // c_o_y)
        i = 0
        crops = lambda img, i, j, s_x, s_y: img[i * s_x : (i+1) * s_x, j * s_y : (j+1) * s_y, :]
        while(i < n_w):
            j = 0
            while(j < n_h):
                x.append(crops(img, i, j, c_x, c_y))
                y.append(crops(img_out, i, j, c_o_x, c_o_y))
                j += 1
            i += 1

        def extend_lists(imgs):
            x.extend(np.array(imgs))
            y.extend(np.array(imgs))
        
        f_a, f_b = __get_img_frame(img, img_x, img_y, img_out, scale)
        x.extend(f_a)
        y.extend(f_b)
    else:
        x.append(img)
        y.append(img_out)
    return x, y


def __get_corresponding_crops(img, img_x, img_y, scale_in, scale_out, scale, scale_down_function):
    img_in, img_out = __get_two_img_scale(img, scale_in, scale_out, scale_down_function)
    return __next_img_crops(img_in, img_out, True, img_x, img_y, scale)

def get_corresponding_crops(img_path, img_x, img_y, scale_in, scale, scale_down_function, img_in_path_perpared = ''):
    scale_out = scale_in / scale
    img = np.array(Image.open(img_path))
    if(img_in_path_perpared != ''):
        img_in = np.array(Image.open(img_in_path_perpared))
        img_out = img    
        if scale_in != 1:
            raise Exception('if you have prepared images, then do not put scale_in for HR image!!!!!')
    else:
        img_in, img_out = __get_two_img_scale(img, scale_in, scale_out, scale_down_function)
    return __next_img_crops(img_in, img_out, True, img_x, img_y, scale)
