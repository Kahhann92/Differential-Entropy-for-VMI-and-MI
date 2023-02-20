import numpy as np
# from scipy.linalg import la
# from keras import optimizers, Sequential, backend
# from keras.utils import to_categorical
# from keras.models import Model
# from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, SpatialDropout2D, ELU, Conv2DTranspose
# from keras.layers import Activation, concatenate, Reshape, BatchNormalization, AveragePooling2D
# from keras.optimizers import SGD
from sklearn.model_selection import RepeatedKFold 
# from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
# from keras import optimizers, Sequential, backend
# from keras.utils import to_categorical
# from keras.models import Model
# from keras.layers import Input, Dense, Flatten, Dropout, ELU
# from keras.layers import Conv3D, SpatialDropout3D, MaxPooling3D, AveragePooling3D
# from keras.layers import Activation, concatenate, Reshape, BatchNormalization, Concatenate
# from keras.optimizers import SGD
from sklearn.model_selection import RepeatedKFold 
# from keras.utils.vis_utils import plot_model
# from keras.models import load_model
import preprocess_methods as pre
import pickle
import os
from tqdm import tqdm
import math
import differential_entropy as de

from sklearn.decomposition import PCA
from sklearn import svm


# import warnings

def scheduler(epoch):
    if epoch < 3:
        return 0.001
    else:
        return 0.001 * pow(0.95,epoch - 10)

def subject_filter(X,y,subjects,choose_sub,method):
    n_data = []
    # print(len(choose_sub))
    # if choose_sub.type == 
    # choose_sub = [choose_sub]
    choose_sub = np.array([choose_sub]).squeeze()
    if choose_sub.shape:
        for subb in choose_sub:
            # print(subb)
            temp = np.where(subjects == subb)
            n_data.extend(temp[0].tolist())
    else:
        n_data = np.where(subjects == choose_sub)
    
    n_data = np.array(n_data)
    X_out = np.take(X,n_data,axis = 0).squeeze()
    
    if method =='meshCNN':
        X_out = np.expand_dims(X_out,4)
    
    y_out = np.take(y,n_data,axis = 0).squeeze()
    subjects_out = subjects[n_data]
#     print(X_out.shape)
    return X_out,y_out,subjects_out

def list2np(data,label,method):
    dataOut, labelOut, SubOut = np.array([]),[],[]
    for subIdx in tqdm(range(len(data)),desc ='Changing to ML data type'):
        temp = data[subIdx]
        y_temp = label[subIdx]
        subject_temp = [subIdx for x in range(temp.shape[0])]

        dataOut = np.concatenate((dataOut, temp),axis= 0) if dataOut.size else temp
        SubOut.extend(subject_temp)
        labelOut.extend(y_temp)
    if method == 'meshCNN':
        dataOut = np.expand_dims(dataOut,axis = 4)
        labelOut = to_categorical(np.array(labelOut) - 1)
    else:
        labelOut = np.array(labelOut) - 1
    # elif method == 'DE':
    #     dataOut = np.expand_dims(dataOut,axis = 4)

    SubOut = np.array(SubOut)
    return dataOut, labelOut, SubOut

def pipe_lines(methods,train_data,train_label,test_data,test_label,norm_state = 'train',supervision_state = 'unsupervised'):
    outputdir = 'model/{method}'.format(method = methods)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    
    if supervision_state =='unsupervised':

        unsupervised_group = []
        unsupervised_group.append(range(7))
        unsupervised_group.append(range(7,14))
        unsupervised_group.append(range(14,len(train_data)))

        
    if methods == 'meshCNN':

        # Step 1 Specialize Preprocess: Crop and mesh
        for subject in tqdm(range(len(train_data)),desc='Preprocessing'):
            start_idx = math.floor(0.5*160)+1
            end_idx = start_idx + 246
            train_data[subject] = train_data[subject][:,:,start_idx:end_idx]
            test_data[subject] = test_data[subject][:,:,start_idx:end_idx]

            train_data[subject],train_label[subject] = pre.crops(train_data[subject],240,train_label[subject])
            test_data[subject],test_label[subject] = pre.crops(test_data[subject],240,test_label[subject])

            train_data[subject] = pre.mesh_2D(train_data[subject])
            test_data[subject] = pre.mesh_2D(test_data[subject])
        train_data, train_label, train_sub = list2np(train_data,train_label,methods)
        test_data, test_label, test_sub = list2np(test_data,test_label,methods)

        # Step 2 normalize
        if norm_state == 'train':
            MEAN_ = train_data.mean()
            STD_ = train_data.std()

            with open('tools/norm_{method}_{time}.pkl'.format(method = methods,time = train_data[0].shape[-1]), 'wb') as f:  
                pickle.dump([MEAN_,STD_], f)
        elif norm_state == 'use':
            with open('tools/norm_{method}_{time}.pkl'.format(method = methods,time = train_data[0].shape[-1]), "rb") as f:  
                MEAN_,STD_ = pickle.load(f)

        train_data = (train_data - MEAN_) / STD_
        test_data = (test_data - MEAN_) / STD_


        callbackss = LearningRateScheduler(scheduler)
            
            
        outputdir = outputdir + '/' + str(supervision_state)
            
        
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        print('Will output at ' + outputdir)

        if supervision_state =='unsupervised':
            verboses, epochs, batch_size = 1, 1,64
            universal_model = initiate_meshed_cnn_model(train_data.shape[1:])
            universal_model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size,
                callbacks=[callbackss],verbose=verboses,validation_data=(test_data, test_label),shuffle=True)

            subject_test_turn = np.unique(test_sub)
            scoresArray = []
            # evaluate subject
            for subject in subject_test_turn:
                test_X, test_y, SUB_test_ = subject_filter(test_data, test_label,test_sub,subject,methods)
                scores = universal_model.evaluate(test_X, test_y, verbose=0)
                print("S%.f,%.2f%%" % (subject, scores[1]*100))
                scoresArray.append(scores[1]*100)

            averageScore = np.average(np.array(scoresArray))
            print("Total Average Score: %.2f%%" % (averageScore))

            fileName = 'universal_model'
            universal_model.save(outputdir + '/'+fileName+'.pkl')

            count = 1
            # unsupervised group
            for group in unsupervised_group:

                train_X, train_y, SUB_train_ = subject_filter(train_data, train_label,test_sub,group)

                model = universal_model
                model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                    callbacks=[callbackss],verbose=verboses,validation_data=(test_data, test_label),shuffle=True)

                # print(value)
                scores = model.evaluate(test_data, test_label, verbose=0)
                print("Group %.f Score: %.2f%%" % (count,scores[1]*100))

                # 保存模型
                fileName = 'model_group_' + str(count)
                model.save(outputdir + '/'+fileName+'.pkl')
                count = count + 1

        else :
            
            universal_path = 'model/meshCNN/unsupervised/universal_model.pkl'
            universal_model = load_model(universal_path)

            verboses, epochs, batch_size = 1, 1,64    
            modelArray = []

            for subject in np.unique(train_sub):
                print('Training for subject ' + str(subject))
                model = universal_model

                train_X, train_y, SUB_test_ = subject_filter(train_data, train_label,train_sub,subject,methods)
                test_X, test_y, SUB_test_ = subject_filter(test_data, test_label,test_sub,subject,methods)

                model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                            callbacks=[callbackss],verbose=verboses,validation_data=(test_X, test_y),shuffle=True)
                modelArray.append(model)


            scoresArray = []
            # evaluate subject
            for subject in np.unique(test_sub):
                
                test_X, test_y, SUB_test_ = subject_filter(train_data, train_label,train_sub,subject)
                scores = modelArray[subject].evaluate(test_X, test_y, verbose=0)
                print("S%.f,%.2f%%" % (subject, scores[1]*100))
                scoresArray.append(scores[1]*100)

            averageScore = np.average(np.array(scoresArray))
            print("Total Average Score: %.2f%%" % (averageScore))
            
            
            subject_no = 0
            
            for model in modelArray:
                fileName = 'sub_{subject_name}_model.pkl'.format(subject_name = subject_no)
                model.save(outputdir + '/'+fileName)
                subject_no = subject_no + 1

    elif methods == 'DE':
        

        # left_chan = [0, 3, 4, 5, 10, 11, 12, 17, 18, 19, 24, 25, 26, 31, 32, 33, 38]
        left_chan = [3,6,7,8,15,16,17,24,25,26,33,34,35,42,43,44,51,52,57,50]
        # right_chan = [2, 9, 8, 7, 16, 15, 14, 23, 22, 21, 30, 29, 28, 37, 36, 35, 40]
        right_chan = [4,12,11,10,21,20,19,30,29,28,39,38,37,48,47,46,55,54,59,56]


        bands = np.array([[8,12],[12,16],[16,20],[20,24],[24,28],[28,35]])
        train_de = []
        test_de = []
        # Step 1 Preprocess
        for subject in tqdm(range(len(train_data)),desc='Preprocessing'):

            # forge diffential entropy
            train_de.append(de.feature_forge(train_data[subject],left_chan,right_chan,bands))
            test_de.append(de.feature_forge(test_data[subject],left_chan,right_chan,bands))

        train_data, train_label, train_sub = list2np(train_de,train_label,methods)
        test_data, test_label, test_sub = list2np(test_de,test_label,methods)
        # %% 
        # Step 2 normalize
        if norm_state == 'train':

            dimension_required = int(train_data.shape[-1]*0.7)
            pca = PCA(n_components=dimension_required)
            pca.fit(train_data)

            with open('tools/pca_{method}.pkl'.format(method = methods,time = train_data[0].shape[-1]), 'wb') as f:  
                pickle.dump(pca, f)
        elif norm_state == 'use':
            with open('tools/pca_{method}.pkl'.format(method = methods,time = train_data[0].shape[-1]), "rb") as f:  
                pca = pickle.load(f)

        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)


        outputdir = outputdir + '/' + str(supervision_state)
            
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        print('Will output at ' + outputdir)
        if supervision_state =='unsupervised':
            
            # universal_model

            clf = svm.SVC(kernel='poly')

            clf.fit(train_data, train_label)
            




            subject_test_turn = np.unique(test_sub)
            scoresArray = []
            # evaluate each single subject
            for subject in subject_test_turn:
                test_X, test_y, SUB_test_ = subject_filter(test_data, test_label,test_sub,subject,methods)
                
                score = clf.score(test_X, test_y)
                print("S%.f,%.2f%%" % (subject, score*100))
                scoresArray.append(score*100)

            averageScore = clf.score(test_data, test_label)
            print("Total Average Score: %.2f%%" % (averageScore))

            modelSet = dict(
                model = clf
            )
            fileName = 'universal_model'
            
            with open(outputdir + '/'+fileName+'.pickle',"wb+") as fp:
                pickle.dump(modelSet,fp,protocol = pickle.HIGHEST_PROTOCOL)

            count = 1


            # unsupervised group
            for group in unsupervised_group:
                train_X, train_y, SUB_train_ = subject_filter(train_data, train_label,test_sub,group,methods)
                clf = svm.SVC(kernel='poly')

                clf.fit(train_X, train_y)
                # print(value)
                score = clf.score(test_data, test_label)
                print("Group %.f Score: %.2f%%" % (count,score))

                # 保存模型
                modelSet = dict(
                    model = clf
                )
                fileName = 'model_group_' + str(count)
                with open(outputdir + '/'+fileName+'.pickle',"wb+") as fp:
                    pickle.dump(modelSet,fp,protocol = pickle.HIGHEST_PROTOCOL)
                count = count + 1

            
        else :
            modelArray = []
            for subject in np.unique(train_sub):
                print('Training for subject ' + str(subject))
                
                train_X, train_y, SUB_test_ = subject_filter(train_data, train_label,train_sub,subject,methods)
                
                clf = svm.SVC(kernel='poly')

                clf.fit(train_X, train_y)
                
                modelArray.append(clf)


            scoresArray = []
            # evaluate subject
            for subject in np.unique(test_sub):
                
                test_X, test_y, SUB_test_ = subject_filter(train_data, train_label,train_sub,subject,methods)

                score = modelArray[subject].score(test_X, test_y)
                
                print("S%.f,%.2f%%" % (subject, score*100))
                scoresArray.append(score*100)

            averageScore = np.average(np.array(scoresArray))
            print("Total Average Score: %.2f%%" % (averageScore))
            
            
            subject_no = 0
            
            for clf in modelArray:
                
                # 保存模型
                modelSet = dict(
                    model = clf
                )
                fileName = 'sub_' + str(subject_no) + '_model'
                with open(outputdir + '/'+fileName+'.pickle',"wb+") as fp:
                    pickle.dump(modelSet,fp,protocol = pickle.HIGHEST_PROTOCOL)
                subject_no = subject_no + 1


# # CSP
# def CSPspatialFilter(Ra,Rb):
#     # Input: Ra,Rb are corvariance matrixs of two classes of data
# 	R = Ra + Rb
# 	E,U = la.eig(R)

# 	# CSP requires the eigenvalues E and eigenvector U be sorted in descending order
# 	ord = np.argsort(E)
# 	ord = ord[::-1] # argsort gives ascending order, flip to get descending
# 	E = E[ord]
# 	U = U[:,ord]

# 	# Find the whitening transformation matrix
# 	P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

# 	# The mean covariance matrices may now be transformed
# 	Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
# 	Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

# 	# Find and sort the generalized eigenvalues and eigenvector
# 	E1,U1 = la.eig(Sa,Sb)
# 	ord1 = np.argsort(E1)
# 	ord1 = ord1[::-1]
# 	E1 = E1[ord1]
# 	U1 = U1[:,ord1]

# 	# The projection matrix (the spatial filter) may now be obtained
# 	SFa = np.dot(np.transpose(U1),P)
# 	return SFa.astype(np.float32)

# # covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
# def covarianceMatrix(A):
# 	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
# 	return Ca

# def CSP(data,label):

#     filters = ()
#     # number of classes    
#     taskNUM = len(np.unique(label))
    
#     for classINx in range(0,taskNUM):
#         # compute covarianceMatrix for class 0
#         # initial
#         class_label = label==classINx
#         R0 = covarianceMatrix(data[class_label][0])
#         for t in range(1,len(data[class_label])):
#             R0 += covarianceMatrix(data[class_label][t])
#             R0 = R0 / len(data[class_label])
#         # compute covarianceMatrix for class 1
#         R1 = covarianceMatrix(data[~class_label][1])
#         for t in range(1,len(data[~class_label])):
#             R1 += covarianceMatrix(data[~class_label][t])
#             R1 = R1 / len(data[~class_label])
#         SFx = CSPspatialFilter(R0,R1)
#         filters += (SFx,)

#     return filters