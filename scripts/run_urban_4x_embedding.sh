source tf/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 
export CUDA_VISIBLE_DEVICES=$1

python3 train.py --dataset ../RDN-Tensorflow/DIV2k --test_dataset ../urban/Urban100 --crop_images True --crop_x_size 64 --crop_y_size 64 --img_x 16 --img_y 16 --scale 4 --test_per_iterations 20000 --steps 2000000 --model_name urban_4x_embedding_img_x_16 --batch_size 16 --data_augment True --embedding_size 32 --learning_rate 6e-05

