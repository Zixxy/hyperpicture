import os
import urllib.request
import shutil
import zipfile

DATA_DIR = './data/'

DIV2K_URL = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
DIV2K_DIR = 'DIV2K_train_HR'

URBAN100_URL = 'https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip'
URBAN100_DIR = 'Urban100'

CIFAR10_URL = 'http://pjreddie.com/media/files/cifar.tgz'
CIFAR10_DIR = 'Cifar10'

SET_5_URL = 'https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip'
SET_5_DIR = 'Set_5'

SET_14_URL = 'https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip'
SET_14_DIR = 'Set_14'


def create_data_dir():
  if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def download_dataset(dataset_url, dataset_name):
  file_name = DATA_DIR + dataset_name

  if not os.path.exists(file_name):
    print("Downloading dataset from url: " + dataset_url)
    with urllib.request.urlopen(dataset_url) as response, open(file_name + '.zip', 'wb') as out_file:
      shutil.copyfileobj(response, out_file)

    print("Unzipping dataset " + file_name + '.zip')
    with zipfile.ZipFile(file_name + '.zip', 'r') as zip_ref:
      zip_ref.extractall(DATA_DIR)

    print("Removing zip file " + file_name + '.zip')
    os.remove(file_name + '.zip')
    print("Dataset downloaded and unzipped at " + file_name)
  else:
    print("Dataset already exists at " + file_name)

create_data_dir()
#download_dataset(DIV2K_URL, DIV2K_DIR)
# download_dataset(URBAN100_URL, URBAN100_DIR)
# download_dataset(SET_5_URL, SET_5_DIR)
# download_dataset(SET_14_URL, SET_14_DIR)
download_dataset(CIFAR10_URL, CIFAR10_DIR)
