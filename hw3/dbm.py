
# coding: utf-8

# # Mean Field Algorithm for DBMs

# In[1]:

import numpy as np
import pdb
import pickle as pkl
import matplotlib
#get_ipython().magic(u'matplotlib inline')
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import argparse
import sys
import os


# In[2]:

## Set random state to reproduce data
rState = np.random.RandomState(212)
# Load Data
train = np.matrix(np.genfromtxt('digitstrain.txt', delimiter=','))
rState.shuffle(train) ## Shuffle to improve convergence
test = np.matrix(np.genfromtxt('digitstest.txt', delimiter=','))
val = np.matrix(np.genfromtxt('digitsvalid.txt', delimiter=','))

# Removing the class from the dataset

_train = train[:,:-1]
_test = test[:,:-1]
_val = val[:,:-1]

# and thresholding to {0,1}
train = np.round(_train)
test = np.round(_test)
val = np.round(_val)


# In[3]:

# Plotting the image 
## The data is in row-major format
def plot_image(train):
  plt.imshow(train[0].reshape((28,28)), cmap=plt.cm.gray)
## The image is squeezed row-wise
#plot_image(train)


# In[4]:

def softplus(X):
  return np.log(1+np.exp(X))

def sigmoid(mat):
  return 1./(1+ np.exp(-mat))

def cross_entropy_loss(vec, gt):
  ## take the average
  return (-np.multiply(gt,np.log(vec)) - np.multiply(1-gt,np.log(1-vec))).sum()/vec.shape[1]

def copy_list(a):
  return [a[i].copy() for i in range(len(a))]


# # Visualization

# In[5]:

## Visualizing filters
def vis(W, save_name):
  dim = W.shape[1]
  n_image_rows = int(np.ceil(np.sqrt(dim)))
  n_image_cols = int(np.ceil(dim * 1.0/n_image_rows))
  gs = gridspec.GridSpec(n_image_rows,n_image_cols,top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
  for g,count in zip(gs,range(int(dim))):
    ax = plt.subplot(g)
    ax.imshow(W[:,count].reshape((28,28)))
    ax.set_xticks([])
    ax.set_yticks([])
  plt.savefig(save_name + '_vis.png')

def plot_cce(model, save_name):  
  train_plt = plt.plot(range(len(model.hist.train_loss)),model.hist.train_loss, 'r--', label='Train')
  val_plt = plt.plot(range(len(model.hist.val_loss)),model.hist.val_loss, 'g-', label="Val")
  plt.xlabel('No. of Epochs')
  plt.ylabel('mean(Entropy Loss)')
  plt.savefig(save_name+'.png')

def plot_err(model, save_name):  
  train_plt = plt.plot(range(len(model.hist.train_loss)),model.hist.train_class_loss, 'r--', label='Train')
  val_plt = plt.plot(range(len(model.hist.val_loss)),model.hist.val_class_loss, 'g-', label="Val")
  plt.xlabel('No. of Epochs')
  plt.ylabel('Classification Error')
  plt.savefig(save_name+'_err.png')


# # Saving and loading the model

# In[65]:

def save_model(model, filename):
  fl = open(filename,'wb')
  pkl.dump(model,fl)
  
def save_weights(model, filename):
  fl = open(filename, 'wb')
  pkl.dump([model.W1, model.W2, model.b, model.c, model.d], fl)
  
def load_model(filename):
  fl = open(filename, 'rb')
  return pkl.load(fl)


# # Loss History Class

# In[66]:

class history(object):
  def __init__(self):
    self.train_loss = list()
    self.val_loss = list()
    
  def add(self,train_loss, val_loss):
    self.train_loss.append(train_loss)
    self.val_loss.append(val_loss)


# # Model Class DBM

# In[173]:

class DBM(object):
  def __init__(self,graph,batch,chains):
    self.graph = graph
    self.hist = history()
    self.chains = chains
    self.batch = batch
    
    ## initialize the weights
    high = np.sqrt(6.0/(sum(graph[:-1])))
    low = -high
    self.W1 = rState.uniform(low=low,high=high, size=(graph[0],graph[1])) # VXH1
    self.W2 = rState.uniform(low=low,high=high, size=(graph[1],graph[2])) # H1XH2
    self.b = rState.uniform(low=low,high=high, size=(graph[0],1)) # VX1
    self.c = rState.uniform(low=low,high=high, size=(graph[1],1)) # H1X1
    self.d = rState.uniform(low=low,high=high, size=(graph[2],1)) # H2X1
    ## E(v,h1,h2) = -v'W1h1 -h1'W2v - b'v - c'h1 - d'h2 
    
    self.randomize_gibbs_updates()
    self.W_optimal = copy_list([self.W1, self.W2, self.b, self.c, self.d])

  def randomize_gibbs_updates(self):
    self.V_tilda = self.sample(rState.uniform(low=0,high=1, size=(self.chains,self.graph[0])))
    self.H1_tilda = self.sample(rState.uniform(low=0,high=1, size=(self.chains,self.graph[1])))
    self.H2_tilda = self.sample(rState.uniform(low=0,high=1, size=(self.chains,self.graph[2])))
    
  def restore_optimal_weights(self):
    self.W1 = self.W_optimal[0]
    self.W2 = self.W_optimal[1]
    self.b = self.W_optimal[2]
    self.c = self.W_optimal[3]
    self.d = self.W_optimal[4]

  def h(self,x):
    return sigmoid(np.matmul(self.W, x) + self.b)
  
  def h_inv(self, h):
    return sigmoid(np.matmul(self.W.T, h) + self.c)
  
  def h1_v_h2(self,v,h2):
    return sigmoid(np.matmul(v, self.W1) + np.matmul(h2, self.W2.T)+ self.c.T)
  
  def v_h1(self, h1):
    return sigmoid(np.matmul(h1, self.W1.T) + self.b.T)
  
  def h2_h1(self, h1):
    return sigmoid(np.matmul(h1, self.W2) + self.d.T)
    
  def sample(self,P):
    return np.random.binomial(1,P,size=P.shape)
  
  ## Mean field updates
  def mfu(self,V):
    self.H1_cap = self.sample(rState.uniform(low=0,high=1, size=(self.batch,self.graph[1])))
    self.H2_cap = self.sample(rState.uniform(low=0,high=1, size=(self.batch,self.graph[2])))
    
    #pdb.set_trace()
    self.H1_cap = sigmoid(np.matmul(V,self.W1) + np.matmul(self.H2_cap, self.W2.T) + self.c.T)
    self.H2_cap = sigmoid(np.matmul(self.H1_cap,self.W2) + self.d.T)
    
  ## MCMC chain to predict a {0,1} output
  def MCMC(self, k):
    ## create a MCMC chain
    for i in range(k):
      _v = sigmoid(np.matmul(self.H1_tilda, self.W1.T) + self.b.T) 
      self.V_tilda = self.sample(_v)
      
      _h2 = sigmoid(np.matmul(self.H1_tilda, self.W2) + self.d.T)
      self.H2_tilda = self.sample(_h2)
      
      _h1 = sigmoid(np.matmul(self.V_tilda, self.W1) + np.matmul(self.H2_tilda, self.W2.T)+ self.c.T)
      self.H1_tilda = self.sample(_h1)
  
  ## MCMC chain to predict a continous output
  def generation(self, x, k):
    _x = self.sample(x)
    _x = _x.T
    _h2 = self.sample(rState.uniform(low=0,high=1,size=(x.shape[1],self.graph[2])))
    #pdb.set_trace()
    for i in tqdm(range(k)):
      _h1 = self.sample(self.h1_v_h2(_x,_h2))
      _h2 = self.sample(self.h2_h1(_h1))
      _h1 = self.sample(self.h1_v_h2(_x,_h2))
      _x = self.sample(self.v_h1(_h1))
    
    return _x.T
  
  def train(self, V, k=1, lr=0.01):
    self.mfu(V)
    self.MCMC(k)
    gradW1 = np.matmul(V.T, self.H1_cap/(self.batch*1.0)) - np.matmul(self.V_tilda.T, self.H1_tilda/(self.chains*1.0)) 
    gradW2 = np.matmul(self.H1_cap.T, self.H2_cap/(self.batch*1.0)) - np.matmul(self.H1_tilda.T, self.H2_tilda/(self.chains*1.0)) 
    gradb = np.mean(V,axis=0) - np.mean(self.V_tilda, axis=0)
    gradc = np.mean(self.H1_cap,axis=0) - np.mean(self.H1_tilda, axis=0)
    gradd = np.mean(self.H2_cap,axis=0) - np.mean(self.H2_tilda, axis=0)
    
    ## gradient updates applied on the weights
    self.W1 += lr*gradW1
    self.W2 += lr*gradW2
    self.b += lr*gradb.T
    self.c += lr*gradc.T
    self.d += lr*gradd.T


# In[174]:

def mean_cross_entropy_loss(model, X, k=1):
  x_gt = X
  _x = model.sample(x_gt)
  _h2 = model.sample(rState.uniform(low=0,high=1,size=(X.shape[0],model.graph[2])))
  for _ in range(k):
    _x = model.sample(_x)
    _h1 = model.sample(model.h1_v_h2(_x,_h2))
    _h2 = model.sample(model.h2_h1(_h1))
    _h1 = model.sample(model.h1_v_h2(_x,_h2))
    _x = model.v_h1(_h1)
    #x_cap = model.h_inv(model.sample(model.h(model.sample(x_gt))))
  l = cross_entropy_loss(_x.T, x_gt.T)
  return l


# In[175]:

def sgd_train(args):
  ## automatically creating a savename
  save_name = "%s_k%d_graph_%d" %(args.save_dir, args.num_mcmc,args.graph[1])
  save_name = os.path.join(args.save_dir, save_name)
  
  ## initilizing the rbm model
  dbm = DBM(args.graph,args.batch, args.chains)
  if args.batch > len(train):
    args.batch = len(train)
  indices = range(0,len(train)+1, args.batch)
  for count in tqdm(range(args.num_epochs)):
    for start,end in zip(indices[:-1], indices[1:]):
      dbm.train(train[start:end,:], k=args.num_mcmc, lr=args.lr)
    ## Calculation of cross entropy loss
    train_loss = mean_cross_entropy_loss(dbm, train)
    val_loss = mean_cross_entropy_loss(dbm, val)
    ## Add loss to the history variable
    dbm.hist.add(train_loss, val_loss)
    tqdm.write("Epochs:%d TrainLoss:%5f ValLoss:%5f" % (count, train_loss, val_loss))
    
  ## plot and save images
  plot_cce(model=dbm, save_name=save_name)
  vis(W=dbm.W1, save_name=save_name)
  
  ## save model
  save_model(dbm, save_name + '_model.p')
  ## save weights
  save_weights(dbm, save_name + 'weights.p')
  return dbm


# In[176]:

def digit_generation(model, filename='temp'):
  X = np.random.uniform(size=train[0:100].T.shape)
  ## binarize the input
  X = np.round(X)
  print np.max(X), np.min(X)
  X_cap = model.generation(X,1000)
  vis(X_cap,filename+'_gen')


# In[177]:

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='save',
                      help='directory to store checkpointed models')
  parser.add_argument('--graph', type=int ,nargs='+', default=[784,100,100],
                      help='Structure of the NN') 
  parser.add_argument('--batch', type=int , default=10,
                      help='mini-batch size') 
  parser.add_argument('--chains', type=int , default=100,
                      help='number of chains') 
  parser.add_argument('--lam', type=float, default=0,
                      help='regularizing coefficient')
  parser.add_argument('--lr', type=float, default=0.01,
                      help='learning rate')
  parser.add_argument('--num_epochs', type=int, default=1, 
                      help='number of epochs')
  parser.add_argument('--num_mcmc', type=int, default=1, 
                      help='number of mcmc iterations')
  parser.add_argument('--seed', type=int, default=212, 
                      help='Random Seed')
  args = parser.parse_args()
  try:
    os.makedirs(args.save_dir)
  except:
    pass
  ## Train model
  model = sgd_train(args)
  
  ## (c) Sample from images
  save_name = "%s_k%d_graph_%d" %(args.save_dir, args.num_mcmc,args.graph[1])
  save_name = os.path.join(args.save_dir, save_name)
  digit_generation(model, filename=save_name)
  return model


# In[ ]:

if __name__=="__main__":
  main()


# In[ ]:



