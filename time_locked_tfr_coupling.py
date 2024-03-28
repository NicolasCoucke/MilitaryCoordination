"""
calculate the individual tfr spectrum, as well as the inter-brain coupling between participants

"""


import io
from copy import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
import sys
import pickle
import os
from my_utils import compute_freq_bands
from mne.time_frequency import tfr_morlet
from scipy.signal import hilbert

import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Assuming signal_1, signal_2, and trf_ppc are predefined and freq is given
# Example: freq = (start_freq, end_freq)

# Function to compute PLV for a given t



sys.path.append('C:/Users/Administrateur/MilitaryCoordination/')


path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1"
raw_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\raw data"
prep_path = os.path.join(path, "time locked preprocessed")
log_path = os.path.join(path, "logs")
connectivity_path = os.path.join(path, "time locked connectivity")

bad_pairs = [12, 13, 22, 26, 42]

import multiprocessing


def calculate_connectivity(pair):
# loop through all data files

    for root, dirs, files in os.walk(prep_path):
        for name in files:
            if 'manual_checked_pair' in name:
                print("processing file " + name)
                file_path = os.path.join(prep_path, name)
                # define paths
                print(file_path)
                
                split_name = name.split("pair_")
                _pair = int(split_name[1])

                if pair != _pair:
                    continue

                if pair in bad_pairs:
                    continue

                with open(file_path,"rb") as input_file:
                    cleaned_epochs_AR = pickle.load(input_file)

            
                
                preproc_S1 = cleaned_epochs_AR[0]
                preproc_S2 = cleaned_epochs_AR[1]

                preproc_S1 = cleaned_epochs_AR[0].interpolate_bads()
                preproc_S2 = cleaned_epochs_AR[1].interpolate_bads()

                #mne.Epochs.plot(preproc_S2, n_channels=64, block = True)
                # define frequency bands 
    
                # select condition and frequency band
                event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}
                pair_complex_signal_dict = {}

                #print(list(preproc_S1.event_id.keys()))
                event_id = preproc_S1.event_id
            # print(event_id)

                # Assuming `event_id` is your dictionary of event IDs
                filtered_event_id = {key: value for key, value in event_id.items() if 'Success/Checkpoint' in key}
                
                # Print the filtered event IDs
                #print(filtered_event_id)


                preproc_S1 = preproc_S1[list(filtered_event_id.keys())]
                preproc_S2 = preproc_S2[list(filtered_event_id.keys())]

                
                # Define frequencies of interest
                freqs = np.arange(1, 45, 1)  # 1 to 40 Hz in 1 Hz steps
                n_cycles = freqs / 4.  # Different number of cycle per frequency

                # Define wavelet parameters
                decim = 2  # To reduce computation time, you can increase this number
                n_jobs = 1  # Number of parallel jobs to run. Can be increased if your machine supports it.

                motor_channels = ["C3", "C1", "Cz", "C2", "C4"]
                frontal_channels = []

            
                #print(np.shape(power_1.data))
                #power_1.plot(picks=['Cz'], baseline=(-0.75, 0.5),  mode='mean', title='MEG 0211 Power');

                """
                power = tfr_morlet(preproc_S2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=decim, n_jobs=n_jobs)    
                power.plot(picks=['Cz'], baseline=(-1, -0.75), mode='mean', title='MEG 0211 Power');

                """


                participant_1_power_values = dict()
                participant_2_power_values = dict()
                # calculate the complex signal (hilbert) for each frequency band 
                power_1_event_id_keys = [str(key).strip() for key in preproc_S1.event_id.keys()]
                
                new_event_ids = []
                for name in filtered_event_id.keys():
                    new_event_ids.append(name.rsplit('/', 1)[0])

            
                power_1 = tfr_morlet(preproc_S1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                    return_itc=False, output = "complex", decim=decim, n_jobs=n_jobs, average = False)    
                    
                power_2 = tfr_morlet(preproc_S2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                    return_itc=False, output = "complex", decim=decim, n_jobs=n_jobs, average = False)    

                for condition in new_event_ids:
            
                    try:
                        # store a version that is averaged over epochs but not yet over channels or time
                        # so it can be used both to plot average scalp power and to plot individual tfr
                        average_powers_1 = np.mean(np.abs(power_1[condition].data)**2, axis = 0)
                        average_powers_2 = np.mean(np.abs(power_2[condition].data)**2, axis = 0)
                    
                    except: 
                        continue
                    
        
                


                    if 'Synchronous/LeaderFollower' in condition:
                        participant_1_power_values['Synchronous/Leader'] = average_powers_1
                        participant_2_power_values['Synchronous/Follower'] = average_powers_2
                    elif 'Synchronous/FollowerLeader' in condition:
                        participant_1_power_values['Synchronous/Follower'] = average_powers_1
                        participant_2_power_values['Synchronous/Leader'] = average_powers_2
                    elif 'Complementary/LeaderFollower' in condition:
                        participant_1_power_values['Complementary/Leader'] = average_powers_1
                        participant_2_power_values['Complementary/Follower'] = average_powers_2
                    elif 'Complementary/FollowerLeader' in condition:
                        participant_1_power_values['Complementary/Follower'] = average_powers_1
                        participant_2_power_values['Complementary/Leader'] = average_powers_2
                    elif 'Synchronous/Egalitarian' in condition:
                        participant_1_power_values['Synchronous/Egalitarian'] = average_powers_1
                        participant_2_power_values['Synchronous/Egalitarian'] = average_powers_2
                    elif 'Complementary/Egalitarian' in condition:
                        participant_1_power_values['Complementary/Egalitarian'] = average_powers_1
                        participant_2_power_values['Complementary/Egalitarian'] = average_powers_2
                    elif 'Individual' in condition:
                        participant_1_power_values['Individual'] = average_powers_1
                        participant_2_power_values['Individual'] = average_powers_2

            
                storepath = os.path.join(connectivity_path, 'individual_tfr_pair'+ str(pair))
                with open(storepath, "wb") as output_file: 
                    pickle.dump([participant_1_power_values, participant_2_power_values], output_file, protocol=pickle.HIGHEST_PROTOCOL)                    
                                





                print('calculate average individual power per condition per participant')
                n_ch = len(preproc_S1.info['ch_names'])

                # calculate many_to_one per participant
                print('calculate many_to_one connectivity per condition per participant')
                biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
                # change the channel names in our epochs so that they are the same as the montage
                channels = biosemi64_montage.ch_names


                
                
                pair_imcoh_values = dict()
                pair_ppc_values = dict()
                pair_trf_synchrony = dict()


                for condition in new_event_ids:
                    try: 

                            

                        signal_1 = power_1[condition].data
                        signal_2 = power_2[condition].data
                    except:
                        # if there is no data for the condition then just go to the next
                        continue

                    # split up into frequencies and calculate for each frequency seperately
                    result_list = []
                    n_epochs, n_ch, n_times =  signal_1.shape[0],  signal_1.shape[1],  signal_1.shape[3]
                    # (2, n_epochs, n_channels, n_freq_bands, n_times)

                    

                    # now calculate the projected power correlations
                    X = signal_1
                    Y = signal_2

                    n_epochs = signal_1.shape[0]
                    n_channels = signal_1.shape[1]
                    n_frequency_bands = signal_1.shape[2]
                    n_times = signal_1.shape[3]

                    # Initialize an array to hold the PPC values for each epoch, channel, and frequency band
                    ppc_over_time = np.zeros((n_epochs, n_channels, n_frequency_bands, n_times))
                    ppc = np.zeros((n_epochs, n_channels, n_frequency_bands))
                    imcoh = np.zeros((n_epochs, n_channels, n_frequency_bands))
                    for epoch in range(n_epochs):
                        for channel in range(n_channels):
                            for i in range(n_frequency_bands):

                                X = signal_1[epoch, channel, i, :]
                                Y = signal_2[epoch, channel, i, :]

                                #IMAGINARY COHERENCE 
                                cross_spectrum = np.mean(X * np.conj(Y), axis=-1)

                                auto_spectrum1 = np.mean(X * np.conj(X), axis=-1)
                                auto_spectrum2 = np.mean(Y * np.conj(Y), axis=-1)

                                # now calculculate the imaginary coherence
                                imag_coherence = np.abs(np.imag(cross_spectrum / np.sqrt(auto_spectrum1 * auto_spectrum2)))


                                # and take the coefficients that belong to the frequency of interest

                                imcoh[epoch, channel, i] = np.mean(imag_coherence)  


                                #PROJECTED POWER CORRELATIONS 
                                X_coeff = X
                                Y_coeff = Y
                                
                                # extract the parts of X and Y that are orthogonal to each other
                                
                                X_orthogonal = np.imag((X_coeff) * np.conj(Y_coeff) / np.abs(Y_coeff))
                                Y_orthogonal = np.imag((Y_coeff) * np.conj(X_coeff) / np.abs(X_coeff))    

                                # get the absolute values of the orthogonal parts
                                
                                X_orthogonal = np.abs(X_orthogonal)
                                Y_orthogonal = np.abs(Y_orthogonal)

                                # average over frequency bands (commented out because now it is per frequency band)
                            # X_orthogonal = np.mean(X_orthogonal, axis = 0) 
                                #Y_orthogonal = np.mean(Y_orthogonal, axis = 0) 




                                # Compute correlation between the magnitudes of the orthogonal projections
                                if np.std(Y_orthogonal) * np.std(X_orthogonal) != 0:  # Avoid division by zero
                                    ppc_value = np.corrcoef(Y_orthogonal, X_orthogonal)[0, 1]
                                else:
                                    ppc_value = 0  # Assign a default value in case of std deviation being zero
                            
                                ppc[epoch, channel, i] = ppc_value


                    # now average over the epochs and store for this pair-condition
                    pair_ppc_values[condition] = np.mean(ppc, axis = 0)
                    pair_imcoh_values[condition] = np.mean(imcoh, axis = 0)


                    # now calculate instantaneous coupling between amplitude enveloppes
                    #n_freq_bands = 4

                    freq_bands = {'Theta': [4, 7],
                                'Alpha': [8, 12],
                                'Beta': [13, 30],
                                'Gamma': [30, 45]}
                    freq_bands = OrderedDict(freq_bands)

                    trf_ppc = np.zeros((n_epochs, n_channels, n_frequency_bands, n_times))
                    for epoch in range(n_epochs):
                        for channel in range(n_channels):
                            for freq in range(len(freqs)):
                                

                                X = signal_1[epoch, channel, freq, :]
                                Y = signal_2[epoch, channel, freq, :]

                                X = np.pad(X, (128, 0), 'constant')
                                Y = np.pad(Y, (128, 0), 'constant')

                                phase_A = np.angle(X)
                                phase_B = np.angle(Y)

                                # Compute the phase difference
                                phase_diff = phase_A - phase_B

                                
                                # Define parameters
                                n = len(phase_diff)
                                window_size = 128
                                num_points = n - window_size  # Adjust for zero-padding

                            # Initialize an empty matrix to hold all windows
                                sub_phase_diff_windows = np.zeros((num_points, window_size))

                                # Fill the matrix with sliding windows of phase_diff
                                for i in range(num_points):
                                    sub_phase_diff_windows[i, :] = phase_diff[i:i+window_size]

                                # Compute PLV for each window
                                exp_values = np.exp(1j * sub_phase_diff_windows)
                                PLV_values = np.abs(np.mean(exp_values, axis=1))

                                trf_ppc[epoch, channel, freq, :num_points] = PLV_values
                    print('condition calculated')

                    # average over trials and store as instantaneous power synchrony measure
                    pair_trf_synchrony[condition] = np.mean(trf_ppc, axis = 0)


                
                storepath = os.path.join(connectivity_path, 'homologous_connectivity_pair_'+ str(pair))
                with open(storepath, "wb") as output_file: 
                    pickle.dump([pair_imcoh_values, pair_ppc_values], output_file, protocol=pickle.HIGHEST_PROTOCOL)


                storepath = os.path.join(connectivity_path, 'homologous_tfr_pair_'+ str(pair))
                with open(storepath, "wb") as output_file: 
                    pickle.dump(pair_trf_synchrony, output_file, protocol=pickle.HIGHEST_PROTOCOL)



try:
    if __name__ == '__main__':
        pool_obj = multiprocessing.Pool(processes = 4)

        pool_obj.map(calculate_connectivity, range(1,45))
except:
    try: 
        if __name__ == '__main__':
            pool_obj = multiprocessing.Pool(processes = 2)

            pool_obj.map(calculate_connectivity, range(1,45))
    except:
        for pair in range(1,45):#[26, 28, 31, 32, 36, 33, 37, 39, 40, 41, 42, 43, 44]:
            calculate_connectivity(pair)