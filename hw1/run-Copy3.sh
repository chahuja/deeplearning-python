#!/bin/bash

#d)
python backprob.py --lr 0.2 --mm 0.5 --save_dir d
python backprob.py --lr 0.2 --mm 0.9 --save_dir d
python backprob.py --lr 0.5 --mm 0.5 --save_dir d
python backprob.py --lr 0.5 --mm 0.9 --save_dir d

