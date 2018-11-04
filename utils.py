import tensorflow as tf
import numpy as np
import scipy
import math

from glob import glob
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

def get_interp_function(function_name):
    if function_name == "cv_bicubic":
        def resize(img, dims):
            return cv2.resize(img, dims, interpolation=cv2.INTER_CUBIC)
        return resize
    if function_name == "scipy":
        def resize(img, dims):
            w,h=dims
            return scipy.misc.imresize(img, (h,w))
        return resize
    if function_name == "scipy_bicubic":
        def resize(img, dims):
            w,h=dims
            return scipy.misc.imresize(img, (h,w), interp='bicubic')
        return resize
    raise Exception('That resize function is not specified')

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def default_tf_variable(shape, scope_name=''):
    if scope_name == '':
        return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05))
    else:
        with tf.name_scope(scope_name):
            var = tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05))
            # var_hist = tf.summary.histogram('deafult_tf_var_hist',  tf.reshape(var, [-1]))
            return var

def batch(iterable, n=32):
    l = len(iterable)
    return [iterable[ndx:min(ndx + n, l)] for ndx in range(0, l, n)]

def batch_gen(iterable, n=32):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)] 

def infinite_batch_gen(iterable, n=32):
    l = len(iterable)
    while True:
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)] 
            
def get_points(x, y):
    return [[i,j] for i in range(x) for j in range(y)]

def show_mnist_img(preds, shape=(28,28)):
    plt.gray()
    plt.figure(figsize=(8, 8))
    plt.imshow(preds.reshape(shape))
    plt.show()
    
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def disp_img(img, upscaled):
    fd = {cPN.X: img.reshape(1, cPN.hparams.img_x, cPN.hparams.img_y, cPN.hparams.channels), 
          cPN.pixels: cPN.points}
    img = sess.run([cPN.logits], feed_dict=fd)

    shape = (cPN.hparams.out_x, cPN.hparams.out_y, cPN.hparams.channels)
    if cPN.hparams.channels == 1: 
        plt.gray()
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(img[0].reshape(shape))
    f.add_subplot(1,2, 2)
    plt.imshow(upscaled.reshape(shape))
    plt.show(block=True)
    
def psnr_and_ssim(img_1, img_2, scale):
    def prepare_img(img):
        img = rgb2ycbcr(img)[:, :, 0:1] / 255.0
        crop = int(scale) + 6
        h,w,_ = img.shape
        img = img[:h - (h % int(scale)), :w - (w % int(scale)), :]
        img = img[crop:-crop, crop:-crop,:]
        return img
    
    img_1 = prepare_img(img_1)
    img_2 = prepare_img(img_2)
    psnr_val = compare_psnr(img_1, img_2)
    ssim_val = compare_ssim(img_1, img_2, win_size=11, gaussian_weights=True, data_range=1.0, multichannel=True, K1=0.01,K2=0.03, sigma=1.5)
    return psnr_val, ssim_val

def run_resize_on_net(resize_fun, imgs, batch_size):
    imgs = np.array(imgs)
    i = 0
    
    res = None
    while(i + batch_size < len(imgs)):
        out = resize_fun(imgs[i:i+batch_size], batch_size)[0]
        if res is None:
            res = out
        else:
            res = np.concatenate((res, out))
        i += batch_size
        
    if i >= len(imgs):
        return res
  
    last_part = imgs[i:] 
    out = resize_fun(last_part, len(last_part))[0]
    if res is None:
        return out
    res = np.concatenate((res, out))
    return res


def conv_layer(x, conv_size, filters_in, filters_out, name, strides = (1,1)):
    with tf.name_scope(name):
        filter_a = default_tf_variable([conv_size, 1, filters_in, filters_out])
        filter_b = default_tf_variable([1, conv_size, filters_out, filters_out])
    x = tf.nn.conv2d(x, filter_a, strides=[1, strides[0], strides[1], 1], padding='SAME')
    x = tf.nn.conv2d(x, filter_b, strides=[1, strides[0], strides[1], 1], padding='SAME')
    return x

def bias(x, bias_size, name):
    with tf.name_scope(name):
        bias = default_tf_variable([bias_size])
    x = tf.nn.bias_add(x, bias)
    return x

def next_layer(x, prev_filters, filters_out, i, layer_size = 3, strides=(1,1)):
    x = conv_layer(x, layer_size, prev_filters, filters_out, 'resnet_layer' + str(i), strides)
    x = bias(x, filters_out, 'resnet_bias' + str(i))
    return x
