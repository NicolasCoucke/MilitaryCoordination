import io
from copy import copy
from collections import OrderedDict
#from hamcrest import none
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import h5py
import mne
#import hypyp
import requests
import os
import PyQt5
import sys
import pickle
#print(sys.path)
sys.path.append('C:/Users/Administrateur/MilitaryCoordination/')

import copy
from copy import deepcopy


path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1"
raw_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\raw data"
prep_path = os.path.join(path, "time locked preprocessed")
log_path = os.path.join(path, "logs")

from autoreject import get_rejection_threshold  # noqa
from mne import create_info
from mne.io.meas_info import _merge_info


# Define the number of contrasts and frequency bands for your grid layout
num_contrasts = 4
num_freq_bands = 4  # Example: Theta, Alpha, Beta, Gamma
# get the montage that we will use
biosemi64_montage = mne.channels.make_standard_montage('biosemi64')

ch_names = biosemi64_montage.ch_names
sfreq = 512  # Example sampling frequency
info1 = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
info1.set_montage(biosemi64_montage)

info2 = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
info2.set_montage(biosemi64_montage)


# loop through all data files
pair = 1
for root, dirs, files in os.walk(prep_path):
    for name in files:
        if ('pair' in name) and ('freq' not in name) and ('manual' in name):
            print("processing file " + name)

            file_path = os.path.join(prep_path, name)
            # define paths
            print(file_path)
            
            split_name = name.split("pair_")
            pair = int(split_name[1])

       
            if pair !=31:
                continue

            with open(file_path,"rb") as input_file:
                cleaned_epochs_AR = pickle.load(input_file)

      

            data1 = cleaned_epochs_AR[0].get_data().copy()
            data2 = cleaned_epochs_AR[1].get_data().copy()

            # Assuming both Epochs objects have the same number of epochs and time points
            n_epochs, n_channels1, n_times = data1.shape
            _, n_channels2, _ = data2.shape

            # Manually create new channel names to avoid conflicts; this is a simplified example
            new_ch_names = [f"{ch}-1" for ch in info1['ch_names']] + [f"{ch}-2" for ch in info2['ch_names']]
            new_n_channels = n_channels1 + n_channels2

            # Create a new info object (simplified approach; adjust according to your needs, including channel types)
            new_info = create_info(ch_names=new_ch_names, sfreq=info1['sfreq'], ch_types=['eeg'] * new_n_channels)


            

            modified_data = np.concatenate((data1, data2), axis=1) 
            visualization_epochs = mne.EpochsArray(modified_data, info=new_info, events=cleaned_epochs_AR[0].events, tmin=cleaned_epochs_AR[0].tmin)

            

            plt.rcParams.update({'font.size': 6})  # Adjust the size as needed

            mne.Epochs.plot(visualization_epochs, n_channels=128, block = True)

            bad_channels_combined = visualization_epochs.info['bads']
            bad_channels_1 = [ch.split('-')[0] for ch in bad_channels_combined if ch.endswith('-1')]
            bad_channels_2 = [ch.split('-')[0] for ch in bad_channels_combined if ch.endswith('-2')]

            cleaned_epochs_AR[0].interpolate_bads()
            cleaned_epochs_AR[1].interpolate_bads()

            cleaned_epochs_AR[0].info['bads'].extend(bad_channels_1)
            cleaned_epochs_AR[1].info['bads'].extend(bad_channels_2)

            # Step 3: Merge bad epochs lists
            combined_bad_epochs = visualization_epochs.drop_log
            rejected_epochs_indices_1 = [i for i, log in enumerate(combined_bad_epochs) if len(log) > 0]


            # Step 4: Reject bad epochs from both Epochs objects
            cleaned_epochs_AR[0].drop(indices=rejected_epochs_indices_1, reason='manual rejection', verbose=True)
            cleaned_epochs_AR[1].drop(indices=rejected_epochs_indices_1, reason='manual rejection', verbose=True)



            # save the data
            storepath = os.path.join(prep_path,"manual_checked_pair_" + str(pair))
           # with open(storepath, "wb") as output_file:
           #     pickle.dump(cleaned_epochs_AR, output_file, protocol=pickle.HIGHEST_PROTOCOL)


            # Step 4: Reject bad epochs from both Epochs objects
           # cleaned_epochs_AR[0].drop(indices=combined_bad_epochs, reason='manual rejection', verbose=True)
            #cleaned_epochs_AR[1].drop(indices=combined_bad_epochs, reason='manual rejection', verbose=True)