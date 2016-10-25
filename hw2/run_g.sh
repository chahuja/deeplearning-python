#!/bin/bash
# (g) run with different hidden variables
python rbm.py --save_dir g --num_epochs 40 --seed 212 --num_mcmc 1 --graph 784 50
python rbm.py --save_dir g --num_epochs 40 --seed 212 --num_mcmc 1 --graph 784 200
python rbm.py --save_dir g --num_epochs 40 --seed 212 --num_mcmc 1 --graph 784 500
