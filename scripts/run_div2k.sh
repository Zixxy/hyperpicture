source tf/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 
export CUDA_VISIBLE_DEVICES=3

python3 train.py --steps 1000000 --test_per_iterations 500 --crop_images True --crop_x_size 160 --crop_y_size 160 --img_x 80 --img_y 80 --out_x 80 --out_y 80
