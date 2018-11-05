mkdir ../data

#cifar10
wget http://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz
mv cifar/ ../data/
rm cifar.tgz
mkdir ../data/cifar/My_outs

#benchmark datasets
wget https://cv.snu.ac.kr/research/EDSR/benchmark.tar #link provided by https://github.com/fperazzi/proSR
tar xzf benchmark.tar
mv benchmark/ ../data/
rm benchmark.tar

