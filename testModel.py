

from mne.filter import filter_data
from mne.io import pick
from mne.io.pick import pick_channels
from numpy.core.defchararray import index, split
from numpy.core.fromnumeric import product, sort
from numpy.lib.arraysetops import unique
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import _num_samples
import classification_methods as methods
from sklearn.metrics import accuracy_score,confusion_matrix
from scipy import signal
from collections import Counter
import pickle
from tqdm import tqdm
from spatialFilter import TRCA,CSP,Xdawn
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import utils
import argparse
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config-0.yaml', help='Path to the config file.')
parser.add_argument('--ResultPath',type=str,default='1', help="output dataset dir")

opts = parser.parse_args()
config = utils.get_config(opts.config)

# 读取训练集
rootdir = 'dataset/WholeSet_source.pickle'
with open(rootdir,"rb") as fp:
    WholeSet = pickle.load(fp)

test_data = WholeSet['test_data']
subNUM = len(test_data)

# 开始测试
supervisions = ['supervised','unsupervised','weak-supervised']

result_directory = os.path.join('results',config['ResultPath'])
if not os.path.exists(result_directory):
    print("Creating directory: {}".format(result_directory))
    os.makedirs(result_directory)
        
for supervisionMethod in supervisions:
    trcaLabelSet = []
    for subINX in tqdm(range(subNUM)):
        # 加载测试数据
        sub_data = test_data[subINX]
        # 加载训练模型
        modeldir = 'trained_model/{supervision}/S{subject}_model.pickle'.format(supervision=supervisionMethod,subject=subINX)
        with open(modeldir,"rb") as fp:
            model = pickle.load(fp)

        # 测试TRCA
        trca_model = model['trca_model']
        trca_model = pickle.loads(trca_model)
        label = trca_model.predict(sub_data)
        trcaLabelSet.append(label)

    
    with open('results/{ResultPath}/{supervision}_results.pickle'.format(ResultPath=config['ResultPath'],supervision=supervisionMethod),"wb+") as fp:
            pickle.dump(trcaLabelSet,fp,protocol = pickle.HIGHEST_PROTOCOL)
      
