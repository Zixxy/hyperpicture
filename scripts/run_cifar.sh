source activate tf
export LD_LIBRARY_PATH=/home/z1102519/storage/conda/miniconda3/envs/tf/lib/
export CUDA_VISIBLE_DEVICES=$1

python3 ../train.py --dataset ../data/cifar/train --scale_in 1 --img_x 32 --img_y 32 --scale 1 --test_dataset ../data/cifar/test --dataset_downscaled ../data/cifar/train --test_dataset_downscaled ../data/cifar/test --test_per_iterations 5000
#--test_per_iterations 5000 
