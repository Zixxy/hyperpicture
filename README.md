# Install Conda
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

# Create a new environment from spec
```
conda create --name networks-do-networks --file conf/spec-file.txt
```

# Activate the env
```
source activate networks-do-networks 
```

# Export required environment variables. NOTE: change a path to Conda to a good one.
```
export LD_LIBRARY_PATH=<PATH_TO_YOUR_CONDA>/conda/miniconda3/envs/tf/lib/ 
export CUDA_VISIBLE_DEVICES=0
```