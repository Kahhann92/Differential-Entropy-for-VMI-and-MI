
# %%
import argparse
import preprocess_methods as pre


def str2bool(v):
    # print (v)
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# %%
# Some arguments parsing:
parser = argparse.ArgumentParser()
# parser.add_argument('--datadir',type=str,default='pool', help="input data dir")
parser.add_argument('--datadir',type=str,default='pool', help="input data dir")
# parser.add_argument('--datadir',type=str,default='source', help="input data dir")
parser.add_argument('--outputdir',type=str,default='dataset', help="output dataset dir")
parser.add_argument('--random_test_index',type=str2bool,nargs='?', const=True, default=True, help="output dataset dir")
# parser.add_argument('--method',type=str,default='meshcnn', help="the classification method you want to use")

opts = parser.parse_args()



# %% Step 1: Read data in mne epoch format and filter data

# WholeSet = pre.readData(rootdir=opts.datadir,base_line_l = 1,motor_imag_l = 3.5, \
#     datasetType = 'korean', NotchFilter = True,BandPassFilter = True,l_freq = 0.5, h_freq = 80,BaseLineRemoval = True)

WholeSet = pre.readData(rootdir=opts.datadir,base_line_l = 0.5,motor_imag_l = 4, \
    datasetType = 'oc', NotchFilter = False,BandPassFilter = False ,l_freq = 0.5, h_freq = 80,BaseLineRemoval = False)


# %% Step 2: Resample & Crop data

WholeSet = pre.resample_n_crop(WholeSet)

# %% Step 3: Average reref

WholeSet = pre.average_reref(WholeSet)
# %% Step 4: Pick Channel & others

train_data, test_data, train_label, test_label = pre.customized_preprocess(WholeSet, opts.datadir,new = opts.random_test_index)

# %% Step 5: Output a pickle dataset for further methods training
pre.save_pickle_loaddataset(train_data, test_data, train_label, test_label, opts.datadir, opts.outputdir)
