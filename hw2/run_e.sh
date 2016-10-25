#!/bin/bash

# (e) Train an autoencoder and use it to initialize a classifier
python autoencoder.py --save_dir e --num_epochs 50 --seed 212 --denoising False
python pretrained.py --save_dir e --num_epochs 50 --seed 212 --weight_file e/e_lr10_mm0_lam0_dropout10_784_100_784weights.p
