import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL

from PIL import Image
from tensorflow.contrib.training import HParams
from tqdm import tqdm

class TargetNetwork:
    def __fc_layers(self, x, weights, biases):
        mul = lambda u: tf.layers.batch_normalization(tf.cos(tf.matmul(x, u[0]) + u[1]), trainable=False)
        x = tf.map_fn(mul, (weights[0], biases[0]), dtype=tf.float32)
        
        with tf.name_scope('target_network'):
            for i in range(1, len(weights)):
                with tf.name_scope('weights_{}'.format(i)):
                    if self.hparams.log_target_weights:
                        tf_w_hist = tf.summary.histogram('weights_hist', tf.reshape(weights[i], [-1]))
                        tf_b_hist = tf.summary.histogram('bias_hist', biases[i])
                    else:
                        print('Not logging target weights')
                        
                    if(i==2):
                        self.later_x = x
                        
                    x = tf.matmul(x, weights[i]) + tf.expand_dims(biases[i],1)
                    
                    if(i < len(weights) - 1):
                        x = tf.cos(x)
                        x = tf.layers.batch_normalization(x, trainable=True)
        return x
    
    def __logits(self, x):
        self.logits = tf.nn.sigmoid(x)

    def __init__(self, hparams, pixels, weights, biases):
        self.hparams = hparams
        x = self.__fc_layers(pixels, weights, biases)
        self.__logits(x)
        