#!/bin/bash

export CARDINALITY=${1}
source ../bin/activate
rm -rf ./runs/
python train_denoising_autoencoder.py --cardinality ${CARDINALITY}\
       --epochs 200 --learning_rate 0.0001 --batch_size 1024
tensorboard --logdir ./runs/ --reload_interval 1
deactivate
