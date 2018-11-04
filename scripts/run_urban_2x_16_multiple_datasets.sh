source activate tf
export LD_LIBRARY_PATH=/home/z1102519/storage/conda/miniconda3/envs/tf/lib/
export CUDA_VISIBLE_DEVICES=$1

python ../train.py --dataset ../data/DIV2K_train_LR_bicubic/DIV2K_train_HR --dataset_downscaled ../data/DIV2K_train_LR_bicubic/x2 --crop_images True --scale_in 1 --img_x 16 --img_y 16 --scale 2 --test_per_iterations 40000 --steps 4000000 --model_name 2x_mgr_solution --batch_size 32 --data_augment True --scale_down_func scipy_bicubic  --test_dataset ../data/benchmark/Urban100/HR ../data/benchmark/Set5/HR ../data/benchmark/B100/HR ../data/benchmark/Set14/HR  --test_dataset_downscaled ../data/benchmark/Urban100/LR_bicubic/X2 ../data/benchmark/Set5/LR_bicubic/X2 ../data/benchmark/B100/LR_bicubic/X2 ../data/benchmark/Set14/LR_bicubic/X2

