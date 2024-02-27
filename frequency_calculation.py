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
import copy
import pickle
import os
#import statsmodels.api as sm
import autoreject
#from statsmodels.formula.api import ols
#print(sys.path)
sys.path.append('C:/Users/Administrateur/MilitaryCoordination/')

# Define your frequency bands
freq_bands = {
    'Theta': [4, 7],
    'Alpha': [8, 12],
    'Beta': [13, 30],
    'Gamma': [30, 45]
}
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


            epoch_envelopes = []
            for epochs in cleaned_epochs_AR:
                envelopes = {}
                for band_name, band_limits in freq_bands.items():
                    # Filter the data for the current frequency band
                    filtered_epochs = epochs.copy().filter(l_freq=band_limits[0], h_freq=band_limits[1], l_trans_bandwidth='auto', h_trans_bandwidth='auto', fir_design='firwin')

                    # maybe later also do it with envelope is false to get the phase of the alphas etc
                    filtered_epochs.apply_hilbert(envelope = False)

                    # Calculate the envelope
                    #envelope = np.abs(analytic_signal)
                    envelopes[band_name] = filtered_epochs
                    epoch_envelopes.append(envelopes)


            # save the data 
            storepath = os.path.join(prep_path,"freq_data_pair_" + str(pair))
            with open(storepath, "wb") as output_file:
                pickle.dump(epoch_envelopes, output_file, protocol=pickle.HIGHEST_PROTOCOL)        


