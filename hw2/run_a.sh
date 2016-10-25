#!/bin/bash
# (a) Different random initializations
python rbm.py --save_dir a1 --num_epochs 40 --seed 100 --num_mcmc 1
python rbm.py --save_dir a2 --num_epochs 40 --seed 200 --num_mcmc 1
python rbm.py --save_dir a3 --num_epochs 40 --seed 300 --num_mcmc 1
python rbm.py --save_dir a4 --num_epochs 40 --seed 400 --num_mcmc 1
python rbm.py --save_dir a5 --num_epochs 40 --seed 500 --num_mcmc 1
