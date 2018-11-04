import tensorflow as tf
import numpy as np

from utils import *

class HyperNetwork:
    def __init__(self, out_size, bias_size, x, hparams, layers):
        self.hparams = hparams
        with tf.name_scope('hyper_network'):
            self.__resnet(x, hparams, layers)
            
    def __resnet(self, x, hparams, layers):
        if self.hparams.in_img_width != self.hparams.in_img_height:
            raise Exception("This model doesn't support different in width and height")
        
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
                                           
        x = tf.reshape(x, shape=[-1, self.hparams.in_img_width, self.hparams.in_img_height, hparams.channels])
        pooling = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        five_five = conv_layer(x, 5, hparams.channels, 10, 'inception_conv_1')
        three_three = conv_layer(x, 3, hparams.channels, 10, 'inception_conv_2')
        one_one = conv_layer(x, 1, hparams.channels, 9, 'inception_conv_3')
        x = tf.concat([one_one, three_three, five_five, pooling], axis=3)
        x = bias(x, 32, 'inception_bias')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        if self.hparams.in_img_width % 8 != 0:
            raise Exception('Size must be diviable by 8')
        iterations = self.hparams.in_img_width / 8
        prev_filters = 32
        j = 0
        for i in range(int(np.log2(iterations))):
            ry = x
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             ry1 = x + ry
#             x = ry1
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             ry2 = x + ry1
#             x = ry2
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
            x = x + ry
#             print(x.shape)
            x = next_layer(x, prev_filters, int(prev_filters*2), j)
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             x = tf.layers.batch_normalization(x)
#             x = tf.nn.relu(x)
            prev_filters = int(prev_filters*2)
#         print(x.get_shape().as_list()[0])
        
        
        y = x
        small_filters = prev_filters
        for i in range(int(prev_filters / 64)):
            y = next_layer(y, small_filters, int(small_filters/2), j)
            y = tf.layers.batch_normalization(y)
            y = tf.nn.relu(y)
            small_filters = int(small_filters/2)

        small_weights_out = y
        small_weights_out = tf.reshape(small_weights_out, [-1, np.prod(small_weights_out.get_shap
        current_out_dim = np.prod(small_weights_out.get_shape().as_list()[1:])
        self.matrices = []
        self.bsss = []
        for i in range(1, len(layers)):
            out_size = layers[i-1] * layers[i]
            bias_size = layers[i]
            if out_size < 8*8*64:
                w,b = self.handle_small_weights(out_size, bias_size, small_weights_out)
            elif out_size <= current_out_dim*2:
                w,b = self.handle_medium_weights(out_size, bias_size, x)
            else:
                w,b = self.handle_big_weights(out_size, bias_size, x)
            w = tf.reshape(w, [-1, layers[i-1], layers[i]])
            b = tf.reshape(b, [-1, layers[i]])
            self.matrices.append(w)
            self.bsss.append(b)
        
    def handle_big_weights(self, out_size, bias_size, x):
        out_weights_prod = np.prod(x.get_shape().as_list()[1:])
        f_out_size = x.get_shape().as_list()[-1]
        i = 0
        b = x
        def output_layers(x, f_in, f_out_size, j):
            x = conv_layer(x, 3, f_in, f_out_size, 'huge_handle_layer' + str(j))
            x = bias(x, f_out_size, 'huge_handle_bias' + str(j))
            return x
        
        x = output_layers(x, f_out_size, f_out_size, i)
        iterations = int(out_size / out_weights_prod)
        for i in range(int(np.log2(iterations))):
            x = output_layers(x, f_out_size, f_out_size*2 ,i+1)
            f_out_size *= 2
        logits = tf.reshape(x, [-1, out_size])

        filters = b.get_shape().as_list()[-1]
        b = conv_layer(b, 3, filters, int(filters/2), 'last_huge_layer_biases' + str(0), strides=(2,2))
        b = bias(b, int(filters/2), 'last_huge_layer_biases_biases' + str(0))
        b = tf.layers.batch_normalization(b)
        b = tf.nn.relu(b)
        
        filters = b.get_shape().as_list()[-1]
        b = conv_layer(b, 3, filters, int(filters/2), 'last_huge_layer_biases' + str(1))
        b = bias(b, int(filters/2), 'last_huge_layer_biases_biases' + str(1))
        b = tf.layers.batch_normalization(b)
        b = tf.nn.relu(b)
        b = tf.reshape(b, [-1, np.prod(b.get_shape().as_list()[1:])])
        out_weights_prod = b.get_shape().as_list()[-1]
        
        with tf.name_scope('last_huge_layer_biases'):
            weights_fc_for_bias = default_tf_variable([out_weights_prod, bias_size], 'weights_fc_huge_for_bias')
            bias_fc_for_bias = default_tf_variable([bias_size], 'bias_fc_huhge_for_bias')
        fc1_for_bias = tf.add(tf.matmul(b, weights_fc_for_bias), bias_fc_for_bias)
        fc1_for_bias = tf.layers.batch_normalization(fc1_for_bias) # maybe remove?
        biases = fc1_for_bias

        return logits, biases
    
    def handle_medium_weights(self, out_size, bias_size, x):
        out_weights_prod = np.prod(x.get_shape().as_list()[1:])
        f_out_size = x.get_shape().as_list()[-1]
        i = 0
        b = x
        def output_layers(x, f_in, f_out_size, j):
            x = conv_layer(x, 3, f_in, f_out_size, 'medium_handle_layer' + str(j))
            x = bias(x, f_out_size, 'medium_handle_bias' + str(j))
#             x = tf.layers.batch_normalization(x)
#             x = tf.nn.relu(x)
            return x
        
        x = output_layers(x, f_out_size, f_out_size, i)
        iterations = int(out_weights_prod / out_size)
        for i in range(int(np.log2(iterations))):
            x = output_layers(x, f_out_size, int(f_out_size/2) ,i+1)
            f_out_size = int(f_out_size/2)
            
        logits = tf.reshape(x, [-1, out_size])
        filters = b.get_shape().as_list()[-1]
        b = conv_layer(b, 3, filters, int(filters/2), 'last_medium_layer_biases' + str(0), strides=(2,2))
        b = bias(b, int(filters/2), 'last_medium_layer_biases_biases' + str(0))
        b = tf.layers.batch_normalization(b)
        b = tf.nn.relu(b)
        
        filters = b.get_shape().as_list()[-1]
        b = conv_layer(b, 3, filters, int(filters/2), 'last_medium_layer_biases' + str(1))
        b = bias(b, int(filters/2), 'last_medium_layer_biases_biases' + str(1))
        b = tf.layers.batch_normalization(b)
        b = tf.nn.relu(b)
        b = tf.reshape(b, [-1, np.prod(b.get_shape().as_list()[1:])])
        out_weights_prod = b.get_shape().as_list()[-1]
        
        
        with tf.name_scope('last_medium_layer_biases'):
            weights_fc_for_bias = default_tf_variable([out_weights_prod, bias_size], 'weights_fc_medium_for_bias')
            bias_fc_for_bias = default_tf_variable([bias_size], 'bias_fc_medium_for_bias')
        fc1_for_bias = tf.add(tf.matmul(b, weights_fc_for_bias), bias_fc_for_bias)
        fc1_for_bias = tf.layers.batch_normalization(fc1_for_bias) # maybe remove?
        biases = fc1_for_bias

        return logits, biases
    
    def handle_small_weights(self, out_size, bias_size, small_weights_out):
        out_weights_prod = np.prod(small_weights_out.get_shape().as_list()[1:])
        with tf.name_scope('fc_layer_weights'):
            weights_fc_for_weights = default_tf_variable([out_weights_prod, out_size], 'weights_fc_for_weights')
            bias_fc_for_weights = default_tf_variable([out_size], 'bias_fc_for_weights')
        fc = tf.add(tf.matmul(small_weights_out, weights_fc_for_weights), bias_fc_for_weights)
        fc = tf.layers.batch_normalization(fc)
        logits = fc
        logits = tf.reshape(logits, [-1, out_size])

        with tf.name_scope('fc_layer_biases'):
            weights_fc_for_bias = default_tf_variable([out_weights_prod, bias_size], 'weights_fc_for_bias')
            bias_fc_for_bias = default_tf_variable([bias_size], 'bias_fc_for_bias')
        fc1_for_bias = tf.add(tf.matmul(small_weights_out, weights_fc_for_bias), bias_fc_for_bias)
        fc1_for_bias = tf.layers.batch_normalization(fc1_for_bias) # maybe remove?
        biases = fc1_for_bias

        return logits, biases
    
    def ramification_layer(self, out_size, bias_size, small_weights_out, big_weights_out):
        BWS = 8 * 8 * 64
        if out_size < BWS:
            with tf.name_scope('fc_layer_weights'):
                weights_fc_for_weights = default_tf_variable([512, out_size], 'weights_fc_for_weights')
                bias_fc_for_weights = default_tf_variable([out_size], 'bias_fc_for_weights')
            fc = tf.add(tf.matmul(small_weights_out, weights_fc_for_weights), bias_fc_for_weights)
            fc = tf.layers.batch_normalization(fc)
            self.logits = fc
        else:
            with tf.name_scope('conv_layer_weights'):
                num_of_iters = out_size // BWS
                res = []
                for i in range(num_of_iters):
                    with tf.name_scope('layer_{}'.format(i)):
                        f1 = default_tf_variable([3, 1, 256, 64], 'f1')
                        f2 = default_tf_variable([1, 3, 64, 64], 'f2')
                    a = tf.nn.conv2d(big_weights_out, f1, strides=[1, 1, 1, 1], padding='SAME')
                    b = tf.nn.conv2d(a, f2, strides=[1, 1, 1, 1], padding='SAME')
                    res.append(b)
                self.logits = tf.concat(res, axis=3)
                self.logits = tf.layers.batch_normalization(self.logits)
        
        logits = tf.reshape(self.logits, [-1, out_size])

        with tf.name_scope('fc_layer_biases'):
            weights_fc_for_bias = default_tf_variable([512, bias_size], 'weights_fc_for_bias')
            bias_fc_for_bias = default_tf_variable([bias_size], 'bias_fc_for_bias')
        fc1_for_bias = tf.add(tf.matmul(small_weights_out, weights_fc_for_bias), bias_fc_for_bias)
        fc1_for_bias = tf.layers.batch_normalization(fc1_for_bias)
        biases = fc1_for_bias

        return logits, biases
