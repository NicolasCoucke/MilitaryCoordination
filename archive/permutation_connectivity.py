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
import pickle
#print(sys.path)
import multiprocessing

sys.path.append('C:/Users/Administrateur/MilitaryCoordination/')

from hypyp import (
    prep,)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import prep
from hypyp import stats
from hypyp import viz
import autoreject

import copy
import my_utils
from my_utils import compute_sync, compute_freq_bands


def calculate_connectivity(pair):
    file_path = os.path.join("F:/hyperscanning_mne", "pair_" + str(pair+1))


    try: 
        with open(file_path , "rb") as input_file:
            cleaned_epochs_AR, dic_AR = pickle.load(input_file)
    except:
        return

    print(cleaned_epochs_AR)
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    freq_bands = {'Theta': [4, 7],
                    'Alpha': [8, 12],
                    'Beta': [13, 30],
                    'Gamma': [30, 45]}
    freq_bands = OrderedDict(freq_bands)
    # select condition and frequency band
    event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}

    pair_complex_signal_dict = {}
    print('calculate complex signal for all conditions')
    for condition in event_id.keys():

        # find the epoch positions of this condition
        epoch_positions = np.where(preproc_S1.events[:,2] == event_id[condition])
        print(epoch_positions)

        if event_id[condition] in preproc_S1.events[:,2]:
            data_inter = np.array([preproc_S1[condition], preproc_S2[condition]])
            complex_signal = compute_freq_bands(data_inter, 512, freq_bands)
            pair_complex_signal_dict[condition] = complex_signal
        else:
            # if there are no epochs for the condition, just continue
            continue


    n_ch = len(preproc_S1.info['ch_names'])
    connectivity_values = dict()
    for condition in event_id.keys():
        complex_signal =   pair_complex_signal_dict[condition]

        # randomly permute the epochs of the first participant of the pair
        complex_signal_1 = complex_signal[0,:,:,:,:]
        n_epochs = np.size(complex_signal,1)
        permutation_index = np.random.permutation(n_epochs)
        complex_signal[0,:,:,:,:] = complex_signal[0,permutation_index,:,:,:]
        print(complex_signal.shape)

        results = compute_sync(complex_signal, mode='wpli', epochs_average = False, save_memory=True)
        connectivity_values[condition] = results

    storepath = os.path.join("F:/hyperscanning_mne", "permuted_connectivity_values_pair_" + str(pair+1))
    with open(storepath, "wb") as output_file: 
        pickle.dump(connectivity_values, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    pair +=1



try:
    if __name__ == '__main__':
        pool_obj = multiprocessing.Pool(processes = 2)

        pool_obj.map(calculate_connectivity, range(23,45))
except:
    try: 
        if __name__ == '__main__':
            pool_obj = multiprocessing.Pool(processes = 2)

            pool_obj.map(calculate_connectivity, range(23,45))
    except:
        for pair in range(23,45):#[26, 28, 31, 32, 36, 33, 37, 39, 40, 41, 42, 43, 44]:
            calculate_connectivity(pair)

calculate_connectivity(1)