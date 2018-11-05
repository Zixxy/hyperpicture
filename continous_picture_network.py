import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import io

from glob import glob
from PIL import Image
from utils import *
from tqdm import tqdm
from hyper_network import HyperNetwork
from target_network import TargetNetwork
from skimage.measure import compare_psnr
from output_image_generator import *

class ContinousPictureNetwork:
    
    def __init__(self, hparams, data_generator, model_name, saved_models_dir):
        self.hparams = hparams
        self.model_name = model_name
        self.saved_models_dir = saved_models_dir
        self.data_generator = data_generator
#         self.__placeholders()
        self.__datasets_inputs()
        self.__define_target_architecture()

        self.img_x, self.img_y = self.hparams.in_img_width, self.hparams.in_img_height
        
        self.logits = self.target_network.logits
        
        self.loss_op = tf.losses.mean_squared_error(self.Y, self.logits)
    
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.hparams.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this trains batch normalziation
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss_op)
    
        self.PSNR = tf.image.psnr(self.logits, self.Y, max_val=1.0)
        self.SSIM = tf.image.ssim(self.logits, self.Y, max_val=1.0)

        tf.summary.scalar("loss", self.loss_op)
        tf.summary.scalar("PSNR", self.PSNR)

        self.merged = tf.summary.merge_all()

        self.__initialize_tf_session()
        self.path_to_psnr = {}

    def __initialize_tf_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
    
    def save(self, step_num):
        print(self.saved_models_dir)
        self.saver.save(self.sess, self.saved_models_dir, step_num)
        
    def restore(self, checkpoint, metagraph):
        new_saver = tf.train.import_meta_graph(metagraph)
        new_saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint))
        
    def __define_target_architecture(self):
        layers = self.hparams.target_layers
#         if self.hparams.random_scales:
#             layers[0] = 3
        if self.hparams.embedding_size > 2:
            layers[0] = self.hparams.embedding_size
            if self.hparams.random_scales:
                raise Exception("Embedding doesn't work with random scales yet.")
        hypernet = HyperNetwork(0, 0, self.X, self.hparams, layers)
        matrices = hypernet.matrices
        biases = hypernet.bsss
        params = {
          'hparams': self.hparams,
          'pixels': self.pixels,
          'weights': matrices,    
          'biases': biases
        }

        self.target_params = params
        self.target_network = TargetNetwork(**params)
        
    def resize_img_with_net(self, img, batch_size, times):
        fd = {self.X: img.reshape(batch_size, self.hparams.in_img_width, self.hparams.in_img_height, self.hparams.channels), 
              self.scale: times}
        img = self.sess.run([self.logits], feed_dict=fd)
        return img
    
    def __embedding_matrix(self):
        in_width = self.hparams.in_img_width
        in_height = self.hparams.in_img_width
        scale_sq = int(self.hparams.scale * self.hparams.scale)
        shape = [self.hparams.in_img_width*self.hparams.in_img_height*scale_sq, self.hparams.embedding_size]
        embedding = default_tf_variable(shape, 'weights_fc_for_weights')
        self.pixels = embedding            
        return in_width, in_height, scale_sq
    
    def __set_pixel_matrix(self):
        in_width = self.hparams.in_img_width
        in_height = self.hparams.in_img_width
        
        pixels_x = tf.linspace(0.0, tf.cast(in_width*self.scale - 1,tf.float32), tf.cast(in_width*self.scale , tf.int32))
        pixels_x = pixels_x * (in_width - 1) / (in_width * self.scale - 1)
        pixels_y = (tf.linspace(0.0, tf.cast(in_height*self.scale - 1,tf.float32), tf.cast(in_height*self.scale , tf.int32))) 
        pixels_y = pixels_y * (in_height - 1) / (in_height * self.scale - 1)

        a, b = pixels_x[ None, :, None ], pixels_y[ :, None, None ]
        cartesian_product = tf.concat( [ a + tf.zeros_like( b ),
                                         tf.zeros_like( a ) + b ], axis = 2 )

        scale_sq = self.scale * self.scale
        self.pixels = tf.reshape(cartesian_product, shape=[tf.cast(in_width * in_height * scale_sq, tf.int32), 2])
        if self.hparams.random_scales:
            filled = tf.fill([tf.cast(in_width * in_height * scale_sq, tf.int32)], self.scale)
            filled = tf.expand_dims(filled, 1)
            self.pixels = tf.concat([self.pixels, filled], axis=1)
            
        return in_width, in_height, scale_sq

    def queue_batch(self, in_image, out_image, scale):
        tensor_list = [in_image, out_image, scale]
        dtypes = [tf.float32, tf.float32, tf.float32]
        shapes = [in_image.get_shape(), out_image.get_shape(), scale.get_shape()]
        q = tf.FIFOQueue(capacity=self.hparams.queue_capacity, dtypes=dtypes)
        enqueue_op = q.enqueue(tensor_list)

        tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op] * self.hparams.queue_threads))
        in_image_batch, out_image_batch, scale_batch = q.dequeue()

        return in_image_batch, out_image_batch, scale_batch
        
    def __datasets_inputs(self):
        img_tensor_shape = tf.TensorShape(
            [None, self.hparams.in_img_width, self.hparams.in_img_height, self.hparams.channels])
        upscaled_img_tensor_shape = tf.TensorShape(
            [None, None, None, self.hparams.channels])
        scale_variable = tf.TensorShape([])
        
        input_dataset = tf.data.Dataset().from_generator(self.data_generator.train_data_generator, 
                                                            output_types=(tf.float32, tf.float32, tf.float32), 
                                                            output_shapes=(img_tensor_shape, upscaled_img_tensor_shape, scale_variable)).repeat()

        self.input_dataset_it = input_dataset.make_one_shot_iterator()
        in_img_tensor, out_img_tensor, scale = self.input_dataset_it.get_next()
        if self.hparams.use_queues:
            in_img_tensor, out_img_tensor, scale = self.queue_batch(in_img_tensor, out_img_tensor, scale)
        
        self.scale = scale
        self.X = in_img_tensor
        if self.hparams.embedding_size > 2:
            in_width, in_height, scale_sq = self.__embedding_matrix()
        else:
            in_width, in_height, scale_sq = self.__set_pixel_matrix()
        
        out_shape = in_width * in_height * scale_sq
        self.Y = tf.reshape(out_img_tensor, 
                            shape=[-1, tf.cast(out_shape, tf.int32), self.hparams.channels])
    
    def __tensorboard_save_images(self, in_img, out_img, tar_img, test_writer, scale, sess, steps):
        def prepare_img(img, fix_dims = True):
            if fix_dims:
                crop = int(scale) + 6
                h,w,_ = img.shape
                img = img[:h - (h % int(scale)), :w - (w % int(scale)), :]
                img = img[crop:-crop, crop:-crop,:]
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.uint8)
            return img
    
        in_img_disp = prepare_img(in_img, False)
        out_img_disp = prepare_img(out_img)
        tar_img_disp = prepare_img(tar_img)
        
        images = np.concatenate([out_img_disp, tar_img_disp])
        image_summary = tf.summary.image("Output and target images. Scale: " + str(scale), images)
        im_sum = sess.run(image_summary)
        test_writer.add_summary(im_sum, steps)
        
        image_summary = tf.summary.image("Input image: " + str(scale), in_img_disp)
        im_sum = sess.run(image_summary)
        test_writer.add_summary(im_sum, steps)
        
    def __save_new_image(self, img, path):
        init_path = path
        path = path.replace('LR_bicubic', 'My_outs')
        if init_path == path:
            path = path.replace('test', 'My_outs')
        if init_path == path:
            raise Exception('The same in and out path while saving examples!')
        
        img = Image.fromarray(img, 'RGB')
        img.save(path)
        
    def __imgs_and_downscaled(self, path, path_downscaled=''):
        imgs = glob(path+ '/*')
        imgs = sorted(imgs)
        if path_downscaled != '':
            imgs_downscaled = glob(path_downscaled+ '/*')
            imgs_downscaled = sorted(imgs_downscaled)
        if(len(imgs) == 0):
            raise Exception("Empty test path directory! " + path + '/*')
        return imgs, imgs_downscaled
    
    def test(self, sess, steps, test_writer, path, scale, path_downscaled=''):
        imgs, imgs_downscaled = self.__imgs_and_downscaled(path, path_downscaled)
        
        def r_test(img_path, img_downscaled):
            return run_test(img_path, 
                            self.resize_img_with_net, 
                            self.hparams.scale_in, 
                            self.hparams.in_img_width, 
                            self.hparams.in_img_height,
                            scale,
                            self.hparams.channels,
                            self.hparams.batch_size,
                            self.hparams.scale_down_func, 
                            img_downscaled)
        
        ssims, psnrs = [], []
        for j in tqdm(range(len(imgs))):
            img_path = imgs[j]
            in_img, tar_img, out_img = r_test(img_path, imgs_downscaled[j])
            in_img = np.clip(in_img, 0, 255)
            tar_img = np.clip(tar_img, 0, 255)
            out_img = np.clip(out_img * 255, 0, 255)
            psnr, ssim = psnr_and_ssim(tar_img.astype(np.uint8), out_img.astype(np.uint8), scale)
            psnrs.append(psnr)
            ssims.append(ssim)

        test_psnr = np.mean(psnrs)
        test_ssim = np.mean(ssims)
        
        if path not in self.path_to_psnr:
            self.path_to_psnr[path] = 0

        if self.path_to_psnr[path] < test_psnr:
            self.path_to_psnr[path] = test_psnr
            for j in tqdm(range(len(imgs))):
                img_path = imgs[j]
                in_img, tar_img, out_img = r_test(img_path, imgs_downscaled[j])
                out_img = np.clip(out_img * 255, 0, 255)
                self.__save_new_image(out_img.astype(np.uint8), imgs_downscaled[j])
        
        test_summary_psnr = tf.Summary(value=[
            tf.Summary.Value(tag="PSNR of test: " + path + " with scale " + str(scale), simple_value=test_psnr),
            tf.Summary.Value(tag="Ssim of test: " + path + " with scale " + str(scale), simple_value=test_ssim),
        ])

        test_writer.add_summary(test_summary_psnr, steps)
        test_writer.flush()
        return test_psnr
        
    def run_test_mode(self, sess, test_writer, test_datasets):
        for test in test_datasets:
            psnr = self.test(sess, 0, test_writer, test, self.hparams.scale, get_mean_res = True)
            print("for test: ", test)
            print("PSNR: ", psnr)
    
    def train(self, sess, train_writer, test_writer):
        tf.train.start_queue_runners(self.sess)

        def run_train(i):
            tensors = [self.merged, self.logits, self.loss_op, self.train_op]
            summary, _, _, _ = sess.run(tensors)
            train_writer.add_summary(summary, i)
            if (i % self.hparams.test_per_iterations == 0) and i>0: # and i > 1000*1000:
#                 self.test(sess, i, test_writer, self.hparams.test_dataset_path)
                self.save(i)
                if self.hparams.random_scales:
                    self.test(sess, i, test_writer, self.hparams.test_dataset_path, 2) #temporily hardcode
                    self.test(sess, i, test_writer, self.hparams.test_dataset_path, 3)
                    self.test(sess, i, test_writer, self.hparams.test_dataset_path, 4)
                else:
                    n = len(self.hparams.test_dataset_path)
                    for k in range(n):
                        test_path = self.hparams.test_dataset_path[k]
                        if len(self.hparams.test_dataset_downscaled) > 0:
                            self.test(sess, i, test_writer, test_path, self.hparams.scale, self.hparams.test_dataset_downscaled[k])
                        else:
                            self.test(sess, i, test_writer, test_path, self.hparams.scale)
        
        if self.hparams.steps <= 0:
            i = 0
            while True:
                run_train(i)
                i += 1
        else:
            for i in tqdm(range(self.hparams.steps)):
                run_train(i)