#!/bin/bash

#e)
python backprob.py --lr 0.01 --mm 0.5 --graph 784 20 10 --save_dir e
python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 10 --save_dir e
python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --save_dir e
python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 10 --save_dir e
