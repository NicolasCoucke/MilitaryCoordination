import io
from copy import copy
from collections import OrderedDict
#from hamcrest import none
import matplotlib.pyplot as plt
import numpy as np
import scipy
import h5py
import mne
import hypyp
import requests
import os
import pickle
import scipy.io as sio

from hypyp import (
    prep,
)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import stats
from hypyp import viz

# load in raw file to use channel locations for plotting
path = r"C:\Users\Administrator\Documents\MATLAB\Hyperscanning Analysis"
filepath = os.path.join(path, "pair_1_sub_epochs.mat")
raw_path = os.path.join(path, "raw data\P1P2_15_02.bdf")


raw_file = mne.io.read_raw_bdf(raw_path, preload = True)#, preload = True)
epochs_object = mne.Epochs(raw_file, np.array([[1, 10, 0]]), preload = True)
channel_names = np.array([ch['ch_name'] for ch in epochs_object.info['chs']])

epochs_object = epochs_object.drop_channels(channel_names[64:])

#epochs_object.info['chs'] = epochs_object.info['chs'][:64]

print(epochs_object.info['chs'])


filepath = os.path.join(path, "python_data\channel_names")
mat_contents = sio.loadmat(filepath)
channels = mat_contents["channels"]
print(channels.shape)
mapping = dict()
for i in range(len(channels[0,:])):
    mapping[epochs_object.info['chs'][i]['ch_name']] = str(channels[0,i][0])
    print(mapping[epochs_object.info['chs'][i]['ch_name']] )
    print(str(channels[0,i][0]))
print(mapping)
mne.rename_channels(epochs_object.info, mapping)


#fieldtrip_epochs = mne.read_epochs_fieldtrip(filepath, raw_file.info , data_name='sub_epoched_data', trialinfo_column=0)
biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
fig = biosemi64_montage.plot(kind='topomap', show = False) 
#plt.show(block=True)
epochs_object.set_montage(biosemi64_montage)

storePath = os.path.join(path, "python_data\plot_cluster_epochs_object.pickle")
with open(storePath, "wb") as output_file: 
   pickle.dump(epochs_object, output_file, protocol=pickle.HIGHEST_PROTOCOL)


