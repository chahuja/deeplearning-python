#!/bin/bash

#python dbm.py --save_dir a1 --graph 784 100 100 --batch 100 --chains 100 --num_epochs 1000 --num_mcmc 1 --seed 100

python dbm.py --save_dir a2 --graph 784 100 100 --batch 100 --chains 100 --num_epochs 1000 --num_mcmc 1 --seed 200

python dbm.py --save_dir a3 --graph 784 100 100 --batch 100 --chains 100 --num_epochs 1000 --num_mcmc 1 --seed 300

python dbm.py --save_dir a4 --graph 784 100 100 --batch 100 --chains 100 --num_epochs 1000 --num_mcmc 1 --seed 400

python dbm.py --save_dir a5 --graph 784 100 100 --batch 100 --chains 100 --num_epochs 1000 --num_mcmc 1 --seed 500

# d
python dbm.py --save_dir d1 --graph 784 200 200 --batch 100 --chains 100 --num_epochs 1000 --num_mcmc 1 --seed 500

# d
python dbm.py --save_dir d2 --graph 784 400 400 --batch 100 --chains 100 --num_epochs 1000 --num_mcmc 1 --seed 500
