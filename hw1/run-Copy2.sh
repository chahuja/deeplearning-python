#!/bin/bash

#d)
python backprob.py --lr 0.1 --mm 0.5 --save_dir d
python backprob.py --lr 0.1 --mm 0.9 --save_dir d
python backprob.py --lr 0.01 --mm 0.5 --save_dir d
python backprob.py --lr 0.01 --mm 0.9 --save_dir d