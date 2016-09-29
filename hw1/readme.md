# Backpropogation in Multilayer perceptorn

## Dependencies
- seaborn (can be installed using pip)
- matplotlib
- numpy

## How to run
The hyper parameters are given as command line arguments. By default these arguments are set to a learning rate of 0.01, a graph of [784,100,10] and other parameters are set to zero.

### Parameters
- --save_dir ; name the directory where you want to save the model and plots
- --graph ; structure of the neural network from input to output
- --prob ; dropout coeffiecient
- --lam ; weight regularization
- --lr ; learning rate
- --mm ; momentum
- --num_epochs ; number of epochs
- --early_stopping ; if you want this set to True

### Examples
The main script is backprob.py
```python
python backprob.py --lr 0.01 --mm 0.5 --graph 784 200 10 --save_dir e
```

Find other examples in run.sh
Please uncomment the line you would like to run.

Also, if you prefer an interactive session, try out the backprob.ipynb which has the same cade as backprob.py but supports inline image display.

Author: Chaitanya Ahuja
Email: cahuja@andrew.cmu.edu
