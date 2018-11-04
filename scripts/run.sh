python3 ../train.py --dataset cifar --img_x 32 --img_y 32 --out_x 32 --out_y 32 --test_per_iterations 100

train.py --dataset ../data/Urban100 --crop_images True --crop_x_size 160 --crop_y_size 160 --img_x 40 --img_y 40 --out_x 40 --out_y 40 --test_per_iterations 500 --test_dataset ../data/Urban100 --learning_rate 5e-05 --steps 1000000