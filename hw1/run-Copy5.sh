#!/bin/bash
#f) Dropout
python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 10 --prob 0.5 --save_dir f
python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --prob 0.5 --save_dir f
python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 10 --prob 0.5 --save_dir f
