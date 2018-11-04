source tf/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 
export CUDA_VISIBLE_DEVICES=$1

python3 train.py --dataset ../RDN-Tensorflow/DIV2k --test_dataset ../urban/Urban100 --crop_images True --scale_in 4 --img_x 16 --img_y 16 --scale 4 --test_per_iterations 20000 --steps 4000000 --model_name different_scales_test --batch_size 16  --random_scales True --data_augment True 

#--metagraph train/saved_models/longer_sota_test_2018_10_06__00_54_26/tmp-480000.meta --checkpoint train/saved_models/longer_sota_test_2018_10_06__00_54_26

