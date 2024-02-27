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
prep_path = os.path.join(path, "preprocessed data")
log_path = os.path.join(path, "logs")



# loop through all data files
pair = 1
for root, dirs, files in os.walk(prep_path):
    for name in files:
        if 'pair' in name:
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

            mne.Epochs.plot(cleaned_epochs_AR[0], n_channels=64, block = True)

            mne.Epochs.plot(cleaned_epochs_AR[1], n_channels=64, block = True)

            print(cleaned_epochs_AR[0].drop_log)

            # Step 2: Retrieve indices of bad epochs
            bad_epochs_0 = [i for i, log in enumerate(cleaned_epochs_AR[0].drop_log) if len(log) != 0]
            bad_epochs_1 = [i for i, log in enumerate(cleaned_epochs_AR[1].drop_log) if len(log) != 0]

            # Step 3: Merge bad epochs lists
            combined_bad_epochs = list(set(bad_epochs_0 + bad_epochs_1))

            # Step 4: Reject bad epochs from both Epochs objects
            cleaned_epochs_AR[0].drop(indices=combined_bad_epochs, reason='manual rejection', verbose=True)
            cleaned_epochs_AR[1].drop(indices=combined_bad_epochs, reason='manual rejection', verbose=True)