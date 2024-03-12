"""
Calculates for each pair the synchrony between homologous electrodes

"""

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
from my_utils import compute_freq_bands
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
prep_path = os.path.join(path, "time locked preprocessed")
log_path = os.path.join(path, "logs")

from scipy.signal import filtfilt, butter

# Create a low-pass filter
nyquist = 0.5 * 512  # Nyquist frequency
cutoff_frequency = 10  # Desired cutoff frequency in Hz
b, a = butter(4, cutoff_frequency / nyquist)


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


            
            preproc_S1 = cleaned_epochs_AR[0]
            preproc_S2 = cleaned_epochs_AR[1]

            # define frequency bands 
            freq_bands = {'Theta': [4, 7],
                            'Alpha': [8, 12],
                            'Beta': [13, 30],
                            'Gamma': [30, 45],
                            'Beta_narrow': [18, 22]}
            freq_bands = OrderedDict(freq_bands)
            # select condition and frequency band
            event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}
            pair_complex_signal_dict = {}

             #print(list(preproc_S1.event_id.keys()))
            event_id = preproc_S1.event_id

            # Assuming `event_id` is your dictionary of event IDs
            filtered_event_id = {key: value for key, value in event_id.items() if 'Success/Checkpoint' in key}

            # Print the filtered event IDs
            #print(filtered_event_id)
            preproc_S1 = preproc_S1[list(filtered_event_id.keys())]
            preproc_S2 = preproc_S2[list(filtered_event_id.keys())]

                
            # calculate the complex signal (hilbert) for each frequency band 
            print('calculate complex signal for all conditions')
            for condition in filtered_event_id.keys():

                # find the epoch positions of this condition
                epoch_positions = np.where(preproc_S1.events[:,2] == event_id[condition])


                if event_id[condition] in preproc_S1.events[:,2]:
  
                    data_inter = np.array([preproc_S1[condition], preproc_S2[condition]])
                    complex_signal = compute_freq_bands(data_inter, 512, freq_bands)
                    pair_complex_signal_dict[condition] = complex_signal
                else:
                    # if there are no epochs for the condition, just continue
                    continue

            print('calculate average individual power per condition per participant')
            n_ch = len(preproc_S1.info['ch_names'])

            participant_1_power_values = dict()
            participant_2_power_values = dict()
            for condition in filtered_event_id.keys():
                try: 
                    complex_signal =   pair_complex_signal_dict[condition]
                except:
                    # if there is no data for the condition then just go to the next
                    continue

                # split up into frequencies and calculate for each frequency seperately
                result_list = []
                n_epochs, n_ch, n_times = complex_signal.shape[1], complex_signal.shape[2], complex_signal.shape[4]
                # (2, n_epochs, n_channels, n_freq_bands, n_times)



            # calculate many_to_one per participant
            print('calculate many_to_one connectivity per condition per participant')
            biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
            # change the channel names in our epochs so that they are the same as the montage
            channels = biosemi64_montage.ch_names


            
            
            pair_imcoh_values = dict()
            pair_ppc_values = dict()
            for condition in filtered_event_id.keys():
                try: 
                    complex_signal =   pair_complex_signal_dict[condition]
                except:
                    # if there is no data for the condition then just go to the next
                    continue

                # split up into frequencies and calculate for each frequency seperately
                result_list = []
                n_epochs, n_ch, n_times = complex_signal.shape[1], complex_signal.shape[2], complex_signal.shape[4]
                # (2, n_epochs, n_channels, n_freq_bands, n_times)

                signal_1 = complex_signal[0, :, :, :, :]
                signal_2 = complex_signal[1, :, :, :, :]


                # now compute the cross_spectrum and auto spectrum of signals (homologous channels)
                # this should remove the time dimension
                cross_spectrum = np.mean(signal_1 * np.conj(signal_2), axis=-1)

                auto_spectrum1 = np.mean(signal_1 * np.conj(signal_1), axis=-1)
                auto_spectrum2 = np.mean(signal_2 * np.conj(signal_2), axis=-1)

                # now calculculate the imaginary coherence
                imag_coherence = np.abs(np.imag(cross_spectrum)) / np.sqrt(auto_spectrum1 * auto_spectrum2)

                # now average across epochs 
                imag_coherence_averaged = np.mean(imag_coherence, axis = 0)
                # so the remaining should be channels x freq bands

                # now calculate the projected power correlations
                X = signal_1
                Y = signal_2

                n_epochs = X.shape[0]
                n_channels = X.shape[1]
                n_frequency_bands = X.shape[2]
                n_times = X.shape[3]

                # Initialize an array to hold the PPC values for each epoch, channel, and frequency band
                ppc = np.zeros((n_epochs, n_channels, n_frequency_bands))

                for epoch in range(n_epochs):
                    for channel in range(n_channels):
                        for frequency_band in range(n_frequency_bands):
                            # Extract the spectral coefficients for the current epoch, channel, and frequency band
                            X_coeff = X[epoch, channel, frequency_band, :]
                            Y_coeff = Y[epoch, channel, frequency_band, :]
                            
                            # Compute the orthogonal projections
                            # Compute the orthogonal projections
                            Y_proj_on_X = np.imag((Y_coeff) * np.conj(X_coeff) / np.abs(X_coeff))
                            X_proj_on_Y = np.imag((X_coeff) * np.conj(Y_coeff) / np.abs(Y_coeff))

                            
                            Y_orthogonal = np.abs(Y_proj_on_X)
                            X_orthogonal = np.abs(X_proj_on_Y)

                            # Apply the filter to the amplitude envelope
                            Y_orthogonal = filtfilt(b, a, Y_orthogonal)
                            X_orthogonal = filtfilt(b, a, X_orthogonal)

                           

                            # Compute correlation between the magnitudes of the orthogonal projections
                            if np.std(Y_orthogonal) * np.std(X_orthogonal) != 0:  # Avoid division by zero
                                ppc_value = np.corrcoef(Y_orthogonal, X_orthogonal)[0, 1]
                            else:
                                ppc_value = 0  # Assign a default value in case of std deviation being zero
                            
                            # Store the PPC value for the current epoch, channel, and frequency band
                            ppc[epoch, channel, frequency_band] = ppc_value

                # now average over the epochs
                ppc_values_averaged = np.mean(ppc, axis = 0)

            
                pair_imcoh_values[condition] = imag_coherence_averaged
                pair_ppc_values[condition] = ppc_values_averaged 

           
            storepath = os.path.join(path, "time locked connectivity", 'homologous_pair_'+ str(pair))
            with open(storepath, "wb") as output_file: 
                pickle.dump([pair_imcoh_values, pair_ppc_values], output_file, protocol=pickle.HIGHEST_PROTOCOL)
