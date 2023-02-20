# ## This is an compliation for several non-training SSVEP classification methods
# Input :signal trial data

from mne.filter import filter_data
from mne.io.pick import pick_channels
from numpy.core.defchararray import index, mod, split
from numpy.core.fromnumeric import product, sort
from numpy.lib.arraysetops import unique
import classification_methods as methods
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
import argparse

import classifier_methods as clss

# 读取训练集 
# 按照不同的监督方式选择训练标签和数据
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config-0.yaml', help='Path to the config file.')
parser.add_argument('--supervision',type=str,default='supervised', help="supervised|weak-supervised|unsupervised")
parser.add_argument('--n_components',type=int,default=5, help="total channel number")
parser.add_argument('--method',type=str,default='DE', help="meshCNN|tfCNN|DE|LSTM,the classification method you want to use")
parser.add_argument('--normalization',type=str,default='train', help="train|use")
# parser.add_argument('--unsupervisedSize',type=int,default=10, help="number of data to train a unsupervised model")

opts = parser.parse_args()
config = opts.config
supervision = opts.supervision
n_components = opts.n_components
methods = opts.method
norm = opts.normalization

# %% read data
if supervision == 'supervised':
    label_loc = 'labelset/StandardLabel_pool.pickle'
    data_loc = 'dataset/WholeSet_pool.pickle'
elif supervision == 'weak-supervised':
    label_loc = 'labelset/weakLabel.pickle'
    data_loc = 'dataset/WholeSet_source.pickle'
else:
    label_loc = 'labelset/StandardLabel_pool.pickle'
    data_loc = 'dataset/WholeSet_pool.pickle'

with open(label_loc,"rb") as fp:
    labelSet = pickle.load(fp)
    train_label = labelSet['train_label']
    test_label = labelSet['test_label']

with open(data_loc,"rb") as fp:
    wholeSet = pickle.load(fp)
    train_data = wholeSet['train_data']
    test_data = wholeSet['test_data']

# %% training begins


clss.pipe_lines(methods,train_data,train_label,test_data,test_label,norm_state = norm,supervision_state = supervision)