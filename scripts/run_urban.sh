source activate ../tf
export LD_LIBRARY_PATH=/home/z1102519/storage/conda/miniconda3/envs/tf/lib/
export CUDA_VISIBLE_DEVICES=$1


python3 train.py --dataset ../RDN-Tensorflow/DIV2k --test_dataset ../urban/Urban100 --crop_images True --crop_x_size 128 --crop_y_size 128 --img_x 32 --img_y 32 --scale 4 --test_per_iterations 20000 --steps 1000000 --model_name newest_model_4_scale --batch_size 16

