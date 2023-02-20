# %%
from mne.filter import filter_data
from mne.io.pick import pick_channels
from numpy.core.defchararray import index, mod, split
from numpy.core.fromnumeric import product, sort
from numpy.lib.arraysetops import unique
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import extract
from sklearn import svm
import classification_methods as methods
from sklearn.metrics import accuracy_score,confusion_matrix
from scipy import signal
from collections import Counter
from mne.preprocessing.xdawn import _fit_xdawn
from imblearn.under_sampling import RandomUnderSampler 
import pickle
import warnings
import scipy.io as scio
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')
# %%
# 读取训练集
# 读取数据
rootdir = 'mini-data'
file_list = os.listdir(path=rootdir)
if '.DS_Store' in file_list:
    file_list.remove('.DS_Store')

WholeSet = []
GoldenLableSet = []
file_list.sort(key = lambda x:x[:-5])
file_list = sorted(file_list)

for filename in tqdm(file_list):
    if '.DS_Store' != filename:
        data_name  = os.path.join(rootdir,filename)
        
        raw_data = scio.loadmat(data_name)
        EEGdata1 = raw_data['EEGdata1']
        EEGdata2 = raw_data['EEGdata2']
        
        trigger = raw_data['trigger_positions']
        label = raw_data['class_labels']


        # 总共有两个block
        epoch_A = methods.Raw2Epoch(EEGdata1,trigger[0],srate=250)
        epoch_B = methods.Raw2Epoch(EEGdata2,trigger[1],srate=250)
        
        # 拼接两个block
        epoch = np.concatenate((epoch_A,epoch_B),axis=0)
        channel_sel = [1,4,6,10,15,18,20,25,27,29,30,35,40,41,42,46,50,55,60]
        epoch_sel = epoch[:,channel_sel,:]
        label = np.concatenate(label,axis=0)

        # data and label
        label = (label==1)*1 #label 转换为0-1

        WholeSet.append(epoch_sel)
        GoldenLableSet.append(label)

#%%
# 分割数据和
# channel_sel = np.linspace(0,channelNUM-1,num=channelNUM-1).astype(int)
# channel_sel = np.delete(channel_sel,33,axis=0)

# repository 作为选择了导联的数据
repository = WholeSet
label_repository = GoldenLableSet
subNUM = len((repository))
epochNUM,channelNUM, Time = np.shape(repository[0])
# 将repository按block分成训练和测试
train_X = repository[::2] #将block1作为训练
train_y = label_repository[::2]
test_X = repository[1::2] #将block2作为测试
test_y = label_repository[1::2]

train_data = []
train_label = []
test_data = []
test_label = []

# 平衡数据类别
for subINX in tqdm(range(int(subNUM/2))):
    # balance train
    train_data_temp = np.reshape(train_X[subINX],(epochNUM,channelNUM*Time))
    rus = RandomUnderSampler(sampling_strategy=0.2)
    train_data_balanced, train_label_balanced = rus.fit_resample(train_data_temp,train_y[subINX])
    train_data.append(np.reshape(train_data_balanced,(train_label_balanced.shape[0],channelNUM,Time)))
    train_label.append(train_label_balanced)
    
    test_data_temp = np.reshape(test_X[subINX],(epochNUM,channelNUM*Time))
    rus = RandomUnderSampler(sampling_strategy=0.2)
    test_data_balanced, test_label_balanced = rus.fit_resample(test_data_temp, test_y[subINX])
    test_data.append(np.reshape(test_data_balanced,(test_data_balanced.shape[0],channelNUM,Time)))
    test_label.append(test_label_balanced)


# %%

trca_svm = svm.SVC()
trca_svm.fit(enhanced_train,sub_label)
predicted = trca_svm.predict(enhanced_test)
trca_svm.score(enhanced_test,test_label[0])

# %%
from spatialFilter import TRCA, Xdawn
xdawn = Xdawn(n_components=6)
xdawn.fit(train_data[0],train_label[0])

# train
train_enhanced = xdawn.transform(train_data[0])
# test
test_enhanced = xdawn.transform(test_data[0])

# %%
train_enhanced.shape

plt.plot(np.mean(train_enhanced[train_label[0]==0],axis=0))
plt.plot(np.mean(train_enhanced[train_label[0]==1],axis=0))

# %%
from sklearn.preprocessing import label_binarize

svc = svm.SVC()

y_train= label_binarize(train_label[0], classes=[0, 1])
y_test = label_binarize(test_label[0], classes=[0, 1])

svc.fit(train_enhanced,y_train)

#%%
from sklearn.preprocessing import label_binarize
y_test = label_binarize(test_label[0], classes=[0, 1])
y_train = label_binarize(test_label[0], classes=[0, 1])

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
y_score = svc.decision_function(test_enhanced)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# %%
from sklearn.metrics import plot_roc_curve

svc_disp = plot_roc_curve(svc, test_enhanced, test_label[0])
plt.show()

# %%
from spatialFilter import CSP
csp = CSP(n_components=10,transform_into='csp_space',cov_est='concat')
csp.fit(train_data[0],train_label[0])
# train
train_enhanced = csp.transform(train_data[0])
# test
test_enhanced = csp.transform(test_data[0])

# %%
plt.plot(np.mean(train_enhanced[train_label[0]==0],axis=0))
plt.plot(np.mean(train_enhanced[train_label[0]==1],axis=0))

# %%
from sklearn.pipeline import Pipeline
from spatialFilter import Xdawn,TRCA,CSP
from sklearn.svm import SVC
pipe_1 = Pipeline([('spatialfilters', Xdawn(n_components=6)), ('svc', SVC())])
pipe_1.fit(train_data[0],train_label[0])
predicted = pipe_1.predict(test_data[0])

pipe_2 = Pipeline([('spatialfilters', CSP(n_components=6,transform_into='csp_space')), ('svc', SVC())])
pipe_2.fit(train_data[0],train_label[0])
predicted = pipe_2.predict(test_data[0])

pipe_3 = Pipeline([('spatialfilters', TRCA(n_components=6)), ('svc', SVC())])
pipe_3.fit(train_data[0],train_label[0])
predicted = pipe_3.predict(test_data[0])

# %%
from sklearn.metrics import plot_roc_curve
svc_disp = plot_roc_curve(pipe_1, test_data[0], test_label[0])
svc_disp = plot_roc_curve(pipe_2, test_data[0], test_label[0])
svc_disp = plot_roc_curve(pipe_3, test_data[0], test_label[0])

plt.show()
# %%
# %%
