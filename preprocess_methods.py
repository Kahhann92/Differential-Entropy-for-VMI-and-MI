import mne
import numpy as np
import os
import scipy.io as scio
from tqdm import tqdm
import pickle


def readData(rootdir,base_line_l = 0.5,motor_imag_l = 3, datasetType = 'korean', NotchFilter = True,BandPassFilter = True,l_freq = 0.5, h_freq = 80,BaseLineRemoval = True):
    """
    rootdir - root directory
    datasetType - currently only tried one
    base_line_l - length of base line to crop
    motor_imag_l - length of motor imaginary to crop
    NotchFilter - to notch filter before epoching, true or false?
    BaseLineRemoval - to remove power at baseline at trial base, true or false?
    """

    # directory location
    data_list = os.listdir(path=rootdir)
    if datasetType == 'korean':
        # parameters
        srate = 512
        channel_name = ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 
            'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
            'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
            'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 
            'FPz', 'FP2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
            'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
            'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
            'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'STIM']

        event_id = dict(left_hand=1, right_hand=2)
        choose_channel = np.arange(64)
        channel_type = ['eeg' for x in range(len(channel_name))]
        channel_type[-1] = 'stim'

        # Initialize an info structure for mne epoch object
        info = mne.create_info(
            ch_names= channel_name,
            ch_types=channel_type,
            sfreq=srate
        )
        
        # %%
        # PS: we have different trials in a mat, so we can only use nested array instead of flat array

        # delete unneccessary file
        if '.DS_Store' in data_list:
            data_list.remove('.DS_Store')
        data_list = sorted(data_list)
        WholeSet = []
        
        subject = 0
        for filename in tqdm(data_list,desc ="Reading data"):
            if filename.split('.')[-1] == 'mat':
                
                path=os.path.join(rootdir, filename)

                # access mat
                mat_content = scio.loadmat(path)

                # access struct
                oct_struct = mat_content['eeg']
                oct_struct.shape
                val = oct_struct[0,0]

                # access struct's value
                imagery_left = np.array(val['imagery_left'])[choose_channel]
                imagery_right = np.array(val['imagery_right'])[choose_channel]
                STI = np.array(val['imagery_event'])

                # construct mne raw array
                STI = np.concatenate((STI, STI *2), axis=1)  # left as 1, right as 2
                raw = np.concatenate((imagery_left, imagery_right), axis=1) 
                raw = np.concatenate((raw, STI), axis=0) 

                raw = mne.io.RawArray(raw,info,verbose = 0)

                if NotchFilter ==True:
                    # notch filter
                    raw.notch_filter(np.arange(60, 241, 60),picks = 'eeg',verbose = 0)

                if BandPassFilter ==True:
                    raw.filter(l_freq, h_freq, verbose=False)
                    # raw.resample(h_freq*2, verbose=False)

                # epoching based on event
                events = mne.find_events(raw, stim_channel='STIM',verbose = 0)

                if BaseLineRemoval == True:
                    epochs = mne.Epochs(raw, events = events, tmin=-1 * base_line_l, event_id = event_id, tmax=motor_imag_l, baseline = (-0.5,-0.2),preload=True,verbose = 0)
                else:
                    epochs = mne.Epochs(raw, events = events, tmin=-1 * base_line_l, event_id = event_id, tmax=motor_imag_l,preload=True,verbose = 0)
                
                epochs.drop_channels('STIM')
                epochs.set_montage('biosemi64', match_case=False, on_missing='raise', verbose=False)

                # append together
                WholeSet.append(epochs)
                
                # next subject
                subject = subject + 1
    elif datasetType == 'oc':
        # parameters
        srate = 1000
        # %%
        # PS: we have different trials in a mat, so we can only use nested array instead of flat array

        # delete unneccessary file
        if '.DS_Store' in data_list:
            data_list.remove('.DS_Store')
        data_list = sorted(data_list)
        WholeSet = []
        
        subject = 0
        for filename in tqdm(data_list,desc ="Reading data"):
            if filename.split('.')[-1] == 'set':
                
                path=os.path.join(rootdir, filename)
                epochs = mne.io.read_epochs_eeglab(path, verbose=False, uint16_codec='latin1')

                if NotchFilter ==True:
                    # notch filter
                    epochs.notch_filter(np.arange(60, 241, 60),picks = 'eeg',verbose = 0)

                if BandPassFilter ==True:
                    epochs.filter(l_freq, h_freq, verbose=False)
                
                WholeSet.append(epochs)
                
                # next subject
                subject = subject + 1
            
            

    
    # WholeSet = np.array(WholeSet)
    
    return WholeSet
    
    

def resample_n_crop(wholeSet,start = 0,end =2):
    for epochs in tqdm(wholeSet,desc = 'Resampling and Cropping data'):
        epochs.resample(160, verbose=False) 
        epochs.crop(-0.5,3,verbose =False) 
        
    return wholeSet

def average_reref(wholeSet):
    [epochs.set_eeg_reference(ref_channels='average',verbose = 0) for epochs in tqdm(wholeSet,desc = 'Average Rereferencing')]
    return wholeSet

def crops(X,required_pnts,Y):
    time_l = X.shape[-1]
    end_idx = [required_pnts]
    start_idx = [0]
    while end_idx[-1] < time_l:
        end_idx.append(end_idx[-1]+1)
        start_idx.append(start_idx[-1]+1)
    # print("Cropping a total of %5d from %5d."% (len(end_idx)*X.shape[0],X.shape[0]))

    X_out = np.array([])
    Y_out = np.array([])
    for i in range(len(end_idx)):
        temp = X[:,:,start_idx[i]:end_idx[i]]
        X_out = np.concatenate((X_out, temp),axis= 0) if X_out.size else temp
        Y_out = np.concatenate((Y_out,Y),axis= 0) if Y_out.size else Y
    return X_out,Y_out

def mesh_2D(X_in):
    X_in = np.transpose(X_in,(1,0,2))
    X_in = np.expand_dims(X_in,axis = 1)
    size_aim = list([7,7])
    size_aim.extend(X_in.shape[-2:]) 
    X_out = np.zeros(size_aim)
    
    skip_list = [0,1,5,6,42,43,47,48]
    count = 0
    read_count = 0
    for i in range(size_aim[0]): # row
        for j in range(size_aim[1]): # column
            if count not in skip_list:
                X_out[i,j,:,:] = X_in[read_count]
                read_count = read_count + 1
            
            count = count + 1 
    X_out = X_out.transpose(2,0,1,3)
    return X_out

def customized_preprocess(wholeSet,datadir,new = False):
    """
    method - meshcnn
    new - if this is the first run
    """
    
    train_data, test_data, train_label, test_label = [],[],[],[]

    # Forge indices for later use
    if new == True:
        pick_valid = 10
        total_idx = []
        for subject in range(len(wholeSet)):
            trials = wholeSet[subject]._data.shape[0]
            Y = wholeSet[subject].events[:,-1]
            idx3 = np.array([])
            for target in np.unique(Y):
                idx1 = np.squeeze(np.where(Y == target))
                idx2 = np.random.choice(idx1, pick_valid, replace=False)
                idx3 = np.concatenate((idx3, idx2),axis= 0) if idx3.size else idx2
            total_idx.append(idx3)
            

        pickle_out = open('tools/{type}_test_indices.pickle'.format(type=datadir),"wb")
        pickle.dump(total_idx, pickle_out)
        pickle_out.close()
        
    else:
        pickle_in = open('tools/{type}_test_indices.pickle'.format(type=datadir),"rb")
        total_idx = pickle.load(pickle_in)
        pickle_in.close()


    for subject in tqdm(range(len(wholeSet)),desc='Separate test set'):
        # get even smaller epoch
        # wholeSet[subject].crop(0,1.535)

        ############# Reorder the channels! 
        # channel_name = wholeSet[subject].ch_names
        # pick_no = [2,36,35,5,4,3,37,38,39,40,8,9,10,46,45,44,43,13,12,11,47,48,49,50,16,17,18,31,55,54,53,21,20,19,30,56,57,58,25,29,62]
        # pick_channel_my = list( channel_name[i] for i in pick_no )
        # # for channel in pick_channel_my:
        # #     print(channel)
        # wholeSet[subject].pick_channels(pick_channel_my,ordered = True)

        # convert to ML readable
        X = wholeSet[subject].get_data()
        Y = wholeSet[subject].events[:,-1]

        # Separate test set
        test_data_ = np.take(X, total_idx[subject], axis=0)
        test_label_ = np.squeeze(np.take(Y,total_idx[subject]))
        train_data_ = np.delete(X,total_idx[subject],0)
        train_label_ = np.delete(Y,total_idx[subject],0)

        # no more super customized

        # # Crops
        # train_data_,train_label_ = crops(train_data_,240,train_label_)
        # test_data_,test_label_ = crops(test_data_,240,test_label_)

        # # Mesh2D
        # train_data_ = mesh_2D(train_data_)
        # test_data_ = mesh_2D(test_data_)

        # add to main list
        train_data.append(train_data_)
        test_data.append(test_data_)
        train_label.append(train_label_)
        test_label.append(test_label_)

    return train_data, test_data, train_label, test_label

        
        
def save_pickle_loaddataset(train_data, test_data, train_label, test_label, datadir, outputdir):
    
    # 保存数据
    dataset = dict(
            train_data = train_data,
            test_data = test_data,
            train_label = train_label,
            test_label = test_label
        )



    with open('{outdir}/WholeSet_{type}.pickle'.format(outdir=outputdir,type=datadir),"wb+") as fp:
        pickle.dump(dataset,fp,protocol = pickle.HIGHEST_PROTOCOL)

    # 保存模型
    labelset = dict(
        train_label = train_label,
        test_label = test_label
    )
    with open('labelset/StandardLabel_{type}.pickle'.format(outdir=outputdir,type=datadir),"wb+") as fp:
        pickle.dump(labelset,fp,protocol = pickle.HIGHEST_PROTOCOL)


