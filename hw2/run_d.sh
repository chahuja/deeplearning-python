#!/bin/bash

# (d) use pretrained weights to initialise the models
python pretrained.py --save_dir d1 --seed 212 --weight_file b/b_k1_graph_100weights.p # k=1 
python pretrained.py --save_dir d2 --seed 212 --weight_file b/b_k5_graph_100weights.p # k=5
python pretrained.py --save_dir d3 --seed 212 --weight_file b/b_k20_graph_100weights.p # k=20
python pretrained.py --save_dir d4 --seed 212 ## random init
