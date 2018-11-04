from glob import glob
from tqdm import tqdm
from PIL import Image
from skimage.color import gray2rgb
from utils import *
from crops_utils import *

import tensorflow as tf
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

def run_resize_on_net(resize_fun, imgs, batch_size, scale):
    imgs = np.array(imgs)
    i = 0
    
    res = None
    while(i + batch_size < len(imgs)):
        out = resize_fun(imgs[i:i+batch_size], batch_size, scale)[0]
        if res is None:
            res = out
        else:
            res = np.concatenate((res, out))
        i += batch_size
        
    if i >= len(imgs):
        return res
  
    last_part = imgs[i:] 
    out = resize_fun(last_part, len(last_part), scale)[0]
    if res is None:
        return out
    res = np.concatenate((res, out))
    return res

def __test_img_with_crops(img, img_out_shape, img_x, img_y, scale, resize_fun, batch_size, channels):
    output = np.zeros(img_out_shape)
    
    w, h, c = img.shape
    o_w, o_h, oc = img_out_shape
    c_x, c_y = img_x, img_y
    c_o_x, c_o_y = int(img_x*scale), int(img_y*scale)
    n_w = w // c_x
    n_h = h // c_y
    i = 0
    crops = lambda img, i, j, s_x, s_y: img[i * s_x : (i+1) * s_x, j * s_y : (j+1) * s_y, :]
    imgs_smaller = [crops(img, i, j, c_x, c_y) for i in range(n_w) for j in range(n_h)]
    all_resized = run_resize_on_net(resize_fun, imgs_smaller, batch_size, scale)
    for i in range(n_w): 
        for j in range(n_h):
            output[i*c_o_x:(i+1)*c_o_x, j*c_o_y:(j+1)*c_o_y, :] = all_resized[i*n_h + j].reshape(c_o_x, c_o_y, channels)# * 255

    frame = []
    begins = [i * c_x for i in range(n_w)]
    begins.append((w - c_x))
    crops = [img[b : b + c_x, h - c_y : h, :] for b in begins]
    bottom_frame = run_resize_on_net(resize_fun, crops, batch_size, scale)
    for i, b in enumerate(begins):
        start = min(int(b*scale), o_w - c_o_x)
        output[start : start + c_o_x, o_h - c_o_y : o_h, :] = bottom_frame[i].reshape(c_o_x, c_o_y, channels)# * 255


    begins = [i * c_y for i in range(n_h)]
    begins.append((h - c_y))
    crops = [img[w - c_x : w, b : b + c_y :] for b in begins]
    right_frame = run_resize_on_net(resize_fun, crops, batch_size, scale)
    for i, b in enumerate(begins):
        start = min(int(b*scale), o_h - c_o_y)
        output[o_w - c_o_x : o_w, start : start + c_o_y :] = right_frame[i].reshape(c_o_x, c_o_y, channels)# * 255
    return output

def __test_on_img(img, img_x, img_y, scale_in, scale_out, scale, resize_fun, batch_size, channels, scale_down_function):
    img_in, img_out = __get_two_img_scale(img, scale_in, scale_out, scale_down_function)
    img_in = img_in / 255
    img_out = img_out
    return img_out, __test_img_with_crops(img_in, img_out.shape, img_x, img_y, scale, resize_fun, batch_size, channels)

def __test_on_img_more_precise(img, img_x, img_y, scale_in, scale_out, scale, resize_fun, batch_size, channels, scale_down_function):
    img_in, img_out = __get_two_img_scale(img, scale_in, scale_out, scale_down_function)
    img_in_2 = np.array(img_in)
    img_in = img_in / 255
    img_out = img_out
    my_out = __compute_average_from_scale_up_crops(img_in, img_x, img_y, scale, resize_fun, batch_size, channels)
    return img_in_2, img_out, my_out

def __compute_average_from_scale_up_crops(small_img, input_x_size, input_y_size, scale, resize_fun, batch_size, channels):
    w,h,c = small_img.shape
    def get_crop_value(decreasing_val, scale_1):
        i = decreasing_val
        while i > 0:
            up = i * scale_1
            if int(up) == up and i < decreasing_val-8:
                break
            i -= 1
        assert i != 0               
        return i
    w_crop = get_crop_value(w, scale)
    h_crop = get_crop_value(h, scale)
    
    img_1 = np.array(small_img[0:w_crop, 0:h_crop,:])
    img_2 = np.array(small_img[w - w_crop:w, 0:h_crop,:])
    img_3 = np.array(small_img[0:w_crop, h-h_crop:h,:])
    img_4 = np.array(small_img[w - w_crop:w, h-h_crop:h,:])
    crops = [img_1, img_2, img_3, img_4]
    w_crop_s = int(w_crop * scale)
    h_crop_s = int(h_crop * scale)
    crop_out_shape = (w_crop_s, h_crop_s, c)
    outs = [__test_img_with_crops(c, crop_out_shape, input_x_size, input_y_size, scale, resize_fun, batch_size, channels) for c in crops]
    
    def make_crops_avarage(crops, w, h, c):
        result = np.zeros([w,h,c])
#         print(result[0:w_crop_s, 0:h_crop_s,:].shape)
#         print(crops[0].shape)
        result[0:w_crop_s, 0:h_crop_s,:] += crops[0]
        result[w - w_crop_s:w, 0:h_crop_s,:] += crops[1]
        result[0:w_crop_s, h - h_crop_s:h,:] += crops[2]
        result[w - w_crop_s:w, h - h_crop_s:h,:] += crops[3]
        result[w - w_crop_s:w_crop_s, h - h_crop_s:h_crop_s,:] /= 4
        result[0:w - w_crop_s, h - h_crop_s:h_crop_s,:] /= 2
        result[w_crop_s:w, h - h_crop_s:h_crop_s,:] /= 2
        result[w - w_crop_s:w_crop_s, 0:h - h_crop_s,:] /= 2
        result[w - w_crop_s:w_crop_s, h_crop_s:h,:] /= 2
        return result
    return make_crops_avarage(outs, int(w*scale), int(h*scale), c)
    
def __fix_greyscale(img):
    if len(img.shape) < 3:
        return gray2rgb(img)
    else:
        return img

def run_test(path, resize_fun, scale_in, input_x_size, input_y_size, scale, channels, batch_size, scale_down_function_name, downscaled_img_path=''):
    scale_down_function = get_interp_function(scale_down_function_name)
    scale_out = scale_in / scale
    img = np.array(Image.open(path))

    if downscaled_img_path != '':
        img_in, img_out = np.array(Image.open(downscaled_img_path)), img
    else:    
        img_in, img_out = __get_two_img_scale(img, scale_in, scale_out, scale_down_function)
    if img_in.shape == img_out.shape and img_in.shape[0] == input_x_size: #MNIST/CIFAR
        my_out = resize_fun(np.expand_dims(img_in /255, axis=0), 1,1)
        return img_in*255, img_out, my_out[0].reshape(img_out.shape)
    img_in, img_out = __fix_greyscale(img_in), __fix_greyscale(img_out)
    img_in_2 = np.array(img_in)
    img_in = img_in / 255
    my_out = __compute_average_from_scale_up_crops(img_in, input_x_size, input_y_size, scale, resize_fun, batch_size, channels)
    return img_in_2, img_out, my_out