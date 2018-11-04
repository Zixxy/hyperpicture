import argparse
import tensorflow as tf
import datetime
import pathlib

from continous_picture_network import ContinousPictureNetwork
from tensorflow.contrib.training import HParams
from lazy_datasets import LazyDataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/DIV2K_train_HR")
parser.add_argument("--dataset_downscaled",default='')

parser.add_argument("--train_dir",default="train")
parser.add_argument("--tensorboard",default="tensorboard")
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--model_name",default="i_will_destroy_humans")
parser.add_argument("--scale_down_func",default="scipy")
parser.add_argument("--checkpoint", default="") 
parser.add_argument("--metagraph", default="") 
parser.add_argument("--learning_rate",default=7e-05, type=float)
parser.add_argument("--steps",default=100000,type=int)
parser.add_argument("--decay_steps",default=100000,type=int)
parser.add_argument("--decay_rate",default=0.92,type=float)

parser.add_argument("--img_x",default=39,type=int)
parser.add_argument("--img_y",default=39,type=int)
parser.add_argument("--scale",default=4,type=float) #tmp
parser.add_argument("--scale_in",default=2,type=float) #tmp
parser.add_argument("--random_scales", default=False,type=bool)
parser.add_argument("--embedding_size",default=0,type=int)

parser.add_argument("--batch_size",default=16,type=int)
parser.add_argument("--channels",default=3,type=int)
parser.add_argument("--log_target_weights",default=False,type=bool)
parser.add_argument("--normalize_colors",default=True,type=bool)
parser.add_argument("--crop_images",default=False,type=bool)
parser.add_argument("--test_per_iterations",default=500, type=int)

parser.add_argument("--queue_capacity",default=32, type=int) 
parser.add_argument("--queue_threads",default=4, type=int)
parser.add_argument("--use_queues",default=True,type=bool)
parser.add_argument("--data_augment",default=False,type=bool)

parser.add_argument('--target_layers', default=[2, 32, 64, 256, 64], nargs='+', type=int)
parser.add_argument('--test_dataset', default=[''], nargs='+')
parser.add_argument('--test_dataset_downscaled', default=[''], nargs='+')

args = parser.parse_args()

target_layers = args.target_layers
target_layers.append(args.channels)

hparams = HParams()
hyper_parameters = {
    'in_img_width': args.img_x,
    'in_img_height': args.img_y,
    'scale': args.scale,
    'crop_images': args.crop_images,
    'scale_in' : args.scale_in,
    'shuffle_data': True,
    'batch_size': args.batch_size,
    'train_dataset_path' : args.dataset,
    'train_dataset_path_downscaled' : args.dataset_downscaled,
    'test_dataset_path' : args.test_dataset,
    'test_dataset_downscaled' : args.test_dataset_downscaled,
    'learning_rate': args.learning_rate,
    'log_target_weights':args.log_target_weights,
    'steps': args.steps,
    'channels': args.channels,
    'random_scales': args.random_scales,
    'normalize_colors': args.normalize_colors,
    'test_per_iterations': args.test_per_iterations,
    'decay_steps':args.decay_steps,
    'decay_rate':args.decay_rate,
    'checkpoint':args.checkpoint,
    'metagraph':args.metagraph,
    'queue_capacity':args.queue_capacity,
    'queue_threads':args.queue_threads,
    'use_queues':args.use_queues,
    'data_augment':args.data_augment,
    'embedding_size':args.embedding_size,
    'target_layers': target_layers,
    'scale_down_func':args.scale_down_func
}
for a,b in hyper_parameters.items():
    hparams.add_hparam(a, b)

# Load data
data_generator = LazyDataset(hparams)

# Prepare train dirs
model_name = args.model_name + '_' + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
print('Running model: ' + model_name)
print('Hyperparameters: ')
for a,b in hyper_parameters.items():
    print(str(a) + ":" + str(b))


pathlib.Path(args.train_dir + '/' + args.savedir).mkdir(parents=True, exist_ok=True) 
saved_models_dir = args.train_dir + '/' + args.savedir + '/' + model_name + '/' + 'tmp'

pathlib.Path(args.train_dir + '/' + args.tensorboard).mkdir(parents=True, exist_ok=True) 
tensorboard_train_dir = args.train_dir + '/' + args.tensorboard + '/' + model_name + '/train'
tensorboard_test_dir = args.train_dir + '/' + args.tensorboard + '/' + model_name + '/test'

# Build net
network = ContinousPictureNetwork(hparams, data_generator, model_name, saved_models_dir)
if hparams.checkpoint != '': 
    print('To restore graph you need to put metagraph path and directorty that contains checkpoint')
    if hparams.metagraph == '':
        raise Exception("Put metagraph path!!!")
    print('restoring checkpoint')
    network.restore(hparams.checkpoint, hparams.metagraph)
    print('restored')
else:
    network.sess.run(network.init)

# Train net
train_writer = tf.summary.FileWriter(tensorboard_train_dir, network.sess.graph)
test_writer = tf.summary.FileWriter(tensorboard_test_dir)
network.train(network.sess, train_writer, test_writer)

