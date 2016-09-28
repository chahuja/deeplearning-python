#!/bin/bash
## 5 for first
##
#d)
python backprob.py --lr 0.1 --save_dir d
python backprob.py --lr 0.01 --save_dir d
python backprob.py --lr 0.2 --save_dir d
python backprob.py --lr 0.5 --save_dir d
python backprob.py --lr 0.1 --mm 0.5 --save_dir d
python backprob.py --lr 0.1 --mm 0.9 --save_dir d
python backprob.py --lr 0.01 --mm 0.5 --save_dir d
python backprob.py --lr 0.01 --mm 0.9 --save_dir d
python backprob.py --lr 0.2 --mm 0.5 --save_dir d
python backprob.py --lr 0.2 --mm 0.9 --save_dir d
python backprob.py --lr 0.5 --mm 0.5 --save_dir d
python backprob.py --lr 0.5 --mm 0.9 --save_dir d

#e)
python backprob.py --lr 0.01 --mm 0.5 --graph 784 20 10 --save_dir e
python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 10 --save_dir e
python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --save_dir e
python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 10 --save_dir e

#f) Dropout
python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 10 --prob 0.5 --save_dir f
python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --prob 0.5 --save_dir f
python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 10 --prob 0.5 --save_dir f

#h) bigger network
python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 100 10 --prob 0.5 --save_dir h
python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 100 10 --prob 0.5 --save_dir h
python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 100 10 --prob 0.5 --save_dir h