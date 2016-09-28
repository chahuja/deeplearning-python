#!/bin/bash
#h) bigger network
python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 100 10 --prob 0.5 --save_dir h
python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 100 10 --prob 0.5 --save_dir h
python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 100 10 --prob 0.5 --save_dir h