"""
Calculates for each participant the power in different frequency bands for all condition

As well as the connectivity with the motor cortes of the other participant

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
from my_utils import compute_freq_bands, compute_freq_bands_memory_saving
from stat_utils import get_two_way_channel_clusters, plot_two_way_channel_clusters
from mne.time_frequency import tfr_morlet
from scipy.signal import hilbert
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

            
            connectivity_path = os.path.join(path, "connectivity data")
            #connectivity_path = os.path.join(path, "connectivity data")
            prep_filename = "".join(['individual_power_pair', str(pair)])
            # if the file already exists then not make it again
            continue_bool = False
            for root, dirs, files in os.walk(connectivity_path):
                if prep_filename in files:
                    continue_bool = True
            if continue_bool:
                continue


            
            preproc_S1 = cleaned_epochs_AR[0]
            preproc_S2 = cleaned_epochs_AR[1]
            del cleaned_epochs_AR

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
            #event_id = preproc_S1.event_id

            

                
            # calculate the complex signal (hilbert) for each frequency band 
            print('calculate complex signal for all conditions')
            for condition in event_id.keys():
                

                if event_id[condition] in preproc_S1.events[:,2]:
                    # find the epoch positions of this condition
                    epoch_positions = np.where(preproc_S1.events[:,2] == event_id[condition])



                    data_inter = np.array([preproc_S1[condition], preproc_S2[condition]])

                     # Define frequencies of interest
                    freqs = np.arange(1, 45, 1)  # 1 to 40 Hz in 1 Hz steps
                    n_cycles = freqs / 4.  # Different number of cycle per frequency

                    # Define wavelet parameters
                    decim = 2  # To reduce computation time, you can increase this number
                    n_jobs = 1  # Number of parallel jobs to run. Can be increased if your machine supports it.

                    # now instead of computing frequency bands, compute the fourier coefficients for all frequencies
                     
                    power_1 = tfr_morlet(preproc_S1[condition], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=decim, n_jobs=n_jobs, average = False)    
                
                    power_2 = tfr_morlet(preproc_S2[condition], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=decim, n_jobs=n_jobs, average = False)    
                    
                    #complex_signal = compute_freq_bands(data_inter, 512, freq_bands)
                    #del data_inter
                    
                    fourrier_coefficients = [power_1, power_2]

                    pair_complex_signal_dict[condition] = fourrier_coefficients
                    #del complex_signal
                else:
                    # if there are no epochs for the condition, just continue
                    continue

            print('calculate average individual power per condition per participant')
            n_ch = len(preproc_S1.info['ch_names'])

            participant_1_power_values = dict()
            participant_2_power_values = dict()
            for condition in event_id.keys():
                try: 
                    fourrier_coefficients =   pair_complex_signal_dict[condition]
                except:
                    # if there is no data for the condition then just go to the next
                    continue
                
                powers = np.zeros((n_epochs, n_channels, n_frequency_bands, n_times))
                # split up into frequencies and calculate for each frequency seperately
                result_list = []
                n_epochs, n_ch, n_times = fourrier_coefficients[0].shape[1], fourrier_coefficients[0].shape[2], fourrier_coefficients[0].shape[4]
                # (2, n_epochs, n_channels, n_freq_bands, n_times)
                
                #magnitude_1 = np.abs(fourrier_coefficients[0])
                #magnitude_2 = np.abs(fourrier_coefficients[1])

                #power_1 = magnitude_1 ** 2
                #power_2 = magnitude_2 ** 2


                
                # final_power (averaged over epochs and times)
                average_powers_1 = np.mean(fourrier_coefficients, axis=(0, 3))
                average_powers_2 = np.mean(fourrier_coefficients, axis=(0, 3))


                if condition == 'Synchronous/LeaderFollower':
                    participant_1_power_values['Synchronous/Leader'] = average_powers_1
                    participant_2_power_values['Synchronous/Follower'] = average_powers_2
                elif condition == 'Synchronous/FollowerLeader':
                    participant_1_power_values['Synchronous/Follower'] = average_powers_1
                    participant_2_power_values['Synchronous/Leader'] = average_powers_2
                elif condition == 'Complementary/LeaderFollower':
                    participant_1_power_values['Complementary/Leader'] = average_powers_1
                    participant_2_power_values['Complementary/Follower'] = average_powers_2
                elif condition == 'Complementary/FollowerLeader':
                    participant_1_power_values['Complementary/Follower'] = average_powers_1
                    participant_2_power_values['Complementary/Leader'] = average_powers_2
                else:
                    participant_1_power_values[condition] = average_powers_1
                    participant_2_power_values[condition] = average_powers_2

            storepath = os.path.join(connectivity_path, 'individual_power_pair'+ str(pair))
            with open(storepath, "wb") as output_file: 
                pickle.dump([participant_1_power_values, participant_2_power_values], output_file, protocol=pickle.HIGHEST_PROTOCOL)    



            # calculate many_to_one per participant
            print('calculate many_to_one connectivity per condition per participant')
            biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
            # change the channel names in our epochs so that they are the same as the montage
            channels = biosemi64_montage.ch_names


            motor_channels = ["C3", "C1", "Cz", "C2", "C4"]
            motor_channel_numbers = motor_channels
            for i in range(len(motor_channels)):
                ch_name = motor_channels[i]
                index = channels.index(ch_name)
                motor_channel_numbers[i] = index


            pair_imcoh_values = []
            pair_ppc_values = []
            
            for many in range(2):
                participant_imcoh_values = dict()
                participant_ppc_values = dict()
                for condition in event_id.keys():
                    try: 
                        complex_signal =   pair_complex_signal_dict[condition]
                    except:
                        # if there is no data for the condition then just go to the next
                        continue

                    # split up into frequencies and calculate for each frequency seperately
                    result_list = []
                    n_epochs, n_ch, n_times = fourrier_coefficients[0].shape[1], fourrier_coefficients[0].shape[2], fourrier_coefficients.shape[4]
                    # (2, n_epochs, n_channels, n_freq_bands, n_times)
                    
                    if many == 0:
                        complex_signal_2 = fourrier_coefficients[1]
                        averaged_motor_signal_2 = np.mean(complex_signal_2[ :, motor_channel_numbers, :, :], axis = 1)
                        n_channels = 64
                        averaged_motor_signal_2 = averaged_motor_signal_2[:, np.newaxis, :, :]
                        #averaged_motor_signal_2= np.tile(averaged_motor_signal_2[:, np.newaxis, :, :], (1, n_channels, 1, 1))
                        signal_2 = averaged_motor_signal_2
                        signal_1 = fourrier_coefficients[0]
                    else:
                        complex_signal_1 = fourrier_coefficients[0]
                        averaged_motor_signal_1 = np.mean(complex_signal_1[ :, motor_channel_numbers, :, :], axis = 1)
                        averaged_motor_signal_1 = averaged_motor_signal_1[:, np.newaxis, :, :]
                        n_channels = 64
                        #averaged_motor_signal_1 = np.tile(averaged_motor_signal_2[:, np.newaxis, :, :], (1, n_channels, 1, 1))
                        signal_1 = averaged_motor_signal_1
                        signal_2 = fourrier_coefficients[1]


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

                   # n_epochs = X.shape[0]
                    n_channels = n_ch
                    n_frequency_bands = len(freq_bands.keys())#X.shape[2]
                    #n_times = X.shape[3]

                    # Initialize an array to hold the PPC values for each epoch, channel, and frequency band
                    ppc = np.zeros((n_epochs, n_channels, n_frequency_bands))

                    for epoch in range(n_epochs):
                        for channel in range(n_channels):
                            for frequency_band in range(n_frequency_bands):
                                # Extract the spectral coefficients for the current epoch, channel, and frequency band
                                if many == 0:
                                    X_coeff = X[epoch, channel, frequency_band, :]
                                    Y_coeff = Y[epoch, 0, frequency_band, :]
                                else:
                                    X_coeff = X[epoch, 0, frequency_band, :]
                                    Y_coeff = Y[epoch, channel, frequency_band, :]
                                
                                # Compute the orthogonal projections
                                Y_proj_on_X = np.imag((Y_coeff) * np.conj(X_coeff) / np.abs(X_coeff))
                                X_proj_on_Y = np.imag((X_coeff) * np.conj(Y_coeff) / np.abs(Y_coeff))

                                
                                Y_orthogonal = np.abs(Y_proj_on_X)
                                X_orthogonal = np.abs(X_proj_on_Y)


                                plt.plot(Y_orthogonal)
                                plt.plot(X_orthogonal)
                                plt.show()

                                # Apply the filter to the amplitude envelope
                                #Y_orthogonal = filtfilt(b, a, Y_orthogonal)
                                #X_orthogonal = filtfilt(b, a, X_orthogonal)


                                # Compute correlation between the magnitudes of the orthogonal projections
                                if np.std(Y_orthogonal) * np.std(X_orthogonal) != 0:  # Avoid division by zero
                                    ppc_value = np.corrcoef(Y_orthogonal, X_orthogonal)[0, 1]
                                else:
                                    ppc_value = 0  # Assign a default value in case of std deviation being zero
                                
                                # Store the PPC value for the current epoch, channel, and frequency band
                                ppc[epoch, channel, frequency_band] = ppc_value

                    # now average over the epochs
                    ppc_values_averaged = np.mean(ppc, axis = 0)

                    if condition == 'Synchronous/LeaderFollower':
                        if many == 0:
                            participant_imcoh_values['Synchronous/Leader'] = imag_coherence_averaged
                            participant_ppc_values['Synchronous/Leader'] = ppc_values_averaged  # Added line for 'ppc'
                        else:
                            participant_imcoh_values['Synchronous/Follower'] = imag_coherence_averaged
                            participant_ppc_values['Synchronous/Follower'] = ppc_values_averaged  # Added line for 'ppc'
                    elif condition == 'Synchronous/FollowerLeader':
                        if many == 1:
                            participant_imcoh_values['Synchronous/Leader'] = imag_coherence_averaged
                            participant_ppc_values['Synchronous/Leader'] = ppc_values_averaged  # Added line for 'ppc'
                        else:
                            participant_imcoh_values['Synchronous/Follower'] = imag_coherence_averaged
                            participant_ppc_values['Synchronous/Follower'] = ppc_values_averaged  # Added line for 'ppc'
                    elif condition == 'Complementary/LeaderFollower':
                        if many == 0:
                            participant_imcoh_values['Complementary/Leader'] = imag_coherence_averaged
                            participant_ppc_values['Complementary/Leader'] = ppc_values_averaged # Added line for 'ppc'
                        else:
                            participant_imcoh_values['Complementary/Follower'] = imag_coherence_averaged
                            participant_ppc_values['Complementary/Follower'] = ppc_values_averaged  # Added line for 'ppc'
                    elif condition == 'Complementary/FollowerLeader':
                        if many == 1:
                            participant_imcoh_values['Complementary/Leader'] = imag_coherence_averaged
                            participant_ppc_values['Complementary/Leader'] = ppc_values_averaged # Added line for 'ppc'
                        else:
                            participant_imcoh_values['Complementary/Follower'] = imag_coherence_averaged
                            participant_ppc_values['Complementary/Follower'] = ppc_values_averaged  # Added line for 'ppc'
                    else:
                        participant_imcoh_values[condition] = imag_coherence_averaged
                        participant_ppc_values[condition] = ppc_values_averaged # Added line for 'ppc'


                pair_imcoh_values.append(participant_imcoh_values)
                pair_ppc_values.append(participant_ppc_values)

            storepath = os.path.join(connectivity_path, 'many_to_one_pair_'+ str(pair))
            with open(storepath, "wb") as output_file: 
                pickle.dump([pair_imcoh_values, pair_ppc_values], output_file, protocol=pickle.HIGHEST_PROTOCOL)
