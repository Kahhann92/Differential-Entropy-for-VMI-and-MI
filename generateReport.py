#%%
import pickle
import numpy as np
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# %%
rootdir = 'results'
labels =[]
file_list = os.listdir(path=rootdir)
if '.DS_Store' in file_list:
    file_list.remove('.DS_Store')
    
for filename in file_list:
    dir = os.path.join(rootdir, filename)
    with open(dir,"rb") as fp:
        label = pickle.load(fp)
    labels.append(label)

labelroot = 'labelSet/StandardLabel_source.pickle'
with open(labelroot,"rb") as fp:
    test_label = pickle.load(fp)
    test_label = test_label['test_label']
# %%
accAll = np.zeros((len(file_list),len(label)))
for methodINX,supervisionMethod in enumerate(file_list):
    # 选择了不同方法的标签
    label = labels[methodINX]
    for subINX in range(len(label)):
        predict_label = label[subINX]
        true_label = test_label[subINX]
        accAll[methodINX,subINX] = accuracy_score(true_label,predict_label)
        
# %%
x = np.linspace(1,accAll.shape[1],accAll.shape[1])
plt.bar(x,accAll[0,:],label=file_list[0],alpha=1)
plt.bar(x,accAll[1,:],label=file_list[1],alpha=0.5)
plt.bar(x,accAll[2,:],label=file_list[2],alpha=0.5)

plt.ylim(0, 1.4)
plt.legend()
plt.show()

# %%
plt.plot(x,accAll[0,:],label=file_list[0])
plt.plot(x,accAll[1,:],label=file_list[1])
plt.plot(x,accAll[2,:],label=file_list[2])
plt.ylim(0, 1.2)

plt.legend()
plt.show()

# %%
