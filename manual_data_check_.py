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




path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1"
raw_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\raw data"
prep_path = os.path.join(path, "time locked preprocessed")
log_path = os.path.join(path, "logs")

from autoreject import get_rejection_threshold  # noqa

# loop through all data files
pair = 1
for root, dirs, files in os.walk(prep_path):
    for name in files:
        if ('pair' in name) and ('freq' not in name):
            print("processing file " + name)

            file_path = os.path.join(prep_path, name)
            # define paths
            print(file_path)
            
            split_name = name.split("_")
            pair = int(split_name[1])

       


            with open(file_path,"rb") as input_file:
                cleaned_epochs_AR = pickle.load(input_file)

            print(np.shape(cleaned_epochs_AR[0].get_data()))
            print(np.shape(cleaned_epochs_AR[1].get_data()))

            

            # We can use the `decim` parameter to only take every nth time slice.
            # This speeds up the computation time. Note however that for low sampling
            # rates and high decimation parameters, you might not detect "peaky artifacts"
            # (with a fast timecourse) in your data. A low amount of decimation however is
            # almost always beneficial at no decrease of accuracy.


            mne.Epochs.plot(cleaned_epochs_AR[0], n_channels=64, block = True)

            mne.Epochs.plot(cleaned_epochs_AR[0], n_channels=64, block = True)
            #print(np.shape(cleaned_epochs_AR[1].get_data()))

            cleaned_epochs_AR[0].interpolate_bads()
            cleaned_epochs_AR[1].interpolate_bads()

            print(cleaned_epochs_AR[0].drop_log)
            bad = list(set(cleaned_epochs_AR[0].drop_log[0].tolist()).union(set(cleaned_epochs_AR[1].drop_log[0].tolist())))
            
            print(bad)


             # Step 2: Retrieve indices of bad epochs
            bad_epochs_0 = [i for i, log in enumerate(cleaned_epochs_AR[0].drop_log) if len(log) != 0]
            bad_epochs_1 = [i for i, log in enumerate(cleaned_epochs_AR[1].drop_log) if len(log) != 0]

            # Step 3: Merge bad epochs lists
            combined_bad_epochs = list(set(bad_epochs_0 + bad_epochs_1))

            # Step 4: Reject bad epochs from both Epochs objects
            cleaned_epochs_AR[0].drop(indices=combined_bad_epochs, reason='manual rejection', verbose=True)
            cleaned_epochs_AR[1].drop(indices=combined_bad_epochs, reason='manual rejection', verbose=True)

           

            # Step 4: Reject bad epochs from both Epochs objects
           # cleaned_epochs_AR[0].drop(indices=combined_bad_epochs, reason='manual rejection', verbose=True)
            #cleaned_epochs_AR[1].drop(indices=combined_bad_epochs, reason='manual rejection', verbose=True)