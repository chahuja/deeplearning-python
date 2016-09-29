#!/bin/bash
# ## 5 for first
python backprob.py --lr 0.01 --save_dir a1
# python backprob.py --lr 0.01 --save_dir a2
# python backprob.py --lr 0.01 --save_dir a3
# python backprob.py --lr 0.01 --save_dir a4
# ##
# #d)
# python backprob.py --lr 0.1 --save_dir d
# python backprob.py --lr 0.01 --save_dir d
# python backprob.py --lr 0.2 --save_dir d
# python backprob.py --lr 0.5 --save_dir d
# python backprob.py --lr 0.1 --mm 0.5 --save_dir d
# python backprob.py --lr 0.1 --mm 0.9 --save_dir d
# python backprob.py --lr 0.01 --mm 0.5 --save_dir d
# python backprob.py --lr 0.01 --mm 0.9 --save_dir d
# python backprob.py --lr 0.2 --mm 0.5 --save_dir d
# python backprob.py --lr 0.2 --mm 0.9 --save_dir d
# python backprob.py --lr 0.5 --mm 0.5 --save_dir d
# python backprob.py --lr 0.5 --mm 0.9 --save_dir d

# #e)
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 20 10 --save_dir e
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 10 --save_dir e
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --save_dir e
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 10 --save_dir e

# #f) Dropout
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 10 --prob 0.5 --save_dir f
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --prob 0.5 --save_dir f
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 10 --prob 0.5 --save_dir f

# #g) best network
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --prob 0.5 --save_dir g --lam 0.1
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --prob 0.5 --save_dir g --lam 0.01
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --prob 0.5 --save_dir g 
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 100 10 --prob 0.5 --save_dir g 

# #h) bigger network
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 100 100 10 --prob 0.5 --save_dir h
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 100 10 --prob 0.5 --save_dir h
# python backprob.py --lr 0.01 --mm 0.5 --graph 784 500 100 10 --prob 0.5 --save_dir h
