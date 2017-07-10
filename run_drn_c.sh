#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.
# train cifar10
#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 128 --num-examples 50000 --gpus=0
#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 128 --num-examples 50000 --gpus=2,3,4,5,6,7

## train resnet-50 
python -u train_drn_c.py --data-dir ~/dataset/ILSVRC/data --data-type imagenet --num-examples 1280019 --frequent 100 --depth 18 --batch-size 256 --gpus=0,1,2,3
