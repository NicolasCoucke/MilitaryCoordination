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
import hypyp
import requests
import os
import PyQt5
import sys
import copy
import pickle
import statsmodels.api as sm
import autoreject
from statsmodels.formula.api import ols
#print(sys.path)
sys.path.append('C:/Users/Administrateur/MilitaryCoordination/')

from hypyp import (
    prep,)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import prep
from hypyp import stats
from hypyp import viz


evoked_desync = []
evoked_sync = []

for pair in range(1,44):

    if (pair) > 20:
        group = 'civilian'
    else:
        group = 'military'

    # here we will load in the preprocessed data to try to get out the erp that we want
    pair = 6
    file_path = os.path.join("F:/hyperscanning_mne/time_locked/", "pair_" + str(pair))     
    with open(file_path , "rb") as input_file:
        cleaned_epochs_AR, dic_AR = pickle.load(input_file)



    # remove the extra rejected channels from the individual trials
    file_path = os.path.join("F:/hyperscanning_mne/time_locked/", "bad_labels_pair_" + str(pair))
    with open(file_path , "rb") as input_file:
        bad_labels = pickle.load(input_file)
    # merge labels
    bad_labels = np.clip(bad_labels[0] + bad_labels[1], 0, 1)
    for cleaned_epochs in cleaned_epochs_AR:
        for epoch in range(np.size(bad_labels, 0)):
            for channel in range(np.size(bad_labels, 1)):
                if bad_labels[epoch, channel] == 0:
                    cleaned_epochs._data[epoch, channel, :] = 0

    epochs_1 = cleaned_epochs_AR[0]
    epochs_2 = cleaned_epochs_AR[1]

    for epochs in [epochs_1, epochs_2]:
        desync_epochs = epochs_1['Desync']
        sync_epochs = epochs_1['Checkpoint']

        
        # Filter the data in the beta band (13-30 Hz)
        desync_epochs_beta = desync_epochs.copy().filter(l_freq=13, h_freq=30)
        sync_epochs_beta = sync_epochs.copy().filter(l_freq=13, h_freq=30)
        # Compute the amplitude envelope
        desync_epochs_beta.apply_hilbert(envelope=True)
        sync_epochs_beta.apply_hilbert(envelope=True)

        # Filter the envelope in the delta band (1-4 Hz)
        desync_epochs_delta = desync_epochs_beta.copy().filter(l_freq=1, h_freq=5)
        sync_epochs_delta = sync_epochs_beta.apply_hilbert(envelope=True)

        desync_epochs_delta.apply_baseline((-1, -0.5))
        erp_desync = desync_epochs_delta.average(picks = [34])

        sync_epochs_delta.apply_baseline((-1, -0.5))
        erp_sync = sync_epochs_delta.average(picks = [34])

        evoked_desync.append(erp_desync)
        evoked_sync.append(erp_sync)


    
    