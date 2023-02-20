import numpy as np
import math
import mne

def differential_entropy_channel(epoch,channels):
    
    whole_data = np.take(epoch,channels,axis=1)

        # crop time
    start_idx = math.floor(0.5*160)+1
    end_idx = start_idx + 400

    whole_data = whole_data[:,:,start_idx:end_idx]

    std_dv = np.std(whole_data,axis=2)
    output = np.log2((2*math.pi*np.exp(1)*np.power(std_dv,2)))/2
    return output

def RASM(epoch,left_channel,right_channel):
    left_DE = differential_entropy_channel(epoch,left_channel)
    right_DE = differential_entropy_channel(epoch,right_channel)
    output = np.divide(left_DE,right_DE) 
    return output


def feature_forge(epoch,left_channel,right_channel,bands):
    DE = np.array([])

    for band in bands:
        temp = mne.filter.filter_data(epoch,160,band[0],band[1],verbose=None)
        temp = RASM(temp,left_channel,right_channel)
        DE = np.concatenate((DE, temp),axis= 1) if DE.size else temp
    return DE