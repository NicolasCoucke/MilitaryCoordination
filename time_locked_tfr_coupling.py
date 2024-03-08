
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
prep_path = os.path.join(path, "time locked preprocessed")
log_path = os.path.join(path, "logs")
connectivity_path = os.path.join(path, "time locked connectivity")


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
           # print(event_id)

            # Assuming `event_id` is your dictionary of event IDs
            filtered_event_id = {key: value for key, value in event_id.items() if 'Success/Checkpoint' in key}

            # Print the filtered event IDs
            #print(filtered_event_id)


            preproc_S1 = preproc_S1[list(filtered_event_id.values())]
            preproc_S2 = preproc_S2[list(filtered_event_id.values())]

            
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

            print(new_event_ids)
            for condition in new_event_ids:
                
                try:

                    power_1 = tfr_morlet(preproc_S1[condition], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=decim, n_jobs=n_jobs, average = False)    
                
                    power_2 = tfr_morlet(preproc_S2[condition], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=decim, n_jobs=n_jobs, average = False)    
                except: 
                    continue
                
    

                average_powers_1 = np.mean(power_1[condition].data, axis = 0)
                average_powers_2 = np.mean(power_2[condition].data, axis = 0)

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
                    participant_2_power_values['Complementary/Egalitarian'] = average_powers_2 # Added line for 'ppc'
                elif 'Individual' in condition:
                    participant_1_power_values['Individual'] = average_powers_1
                    participant_2_power_values['Individual'] = average_powers_2# Added line for 'ppc'

        
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
            for condition in filtered_event_id.keys():
                try: 

                    power_1 = tfr_morlet(preproc_S1[condition], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=decim, n_jobs=n_jobs, average = False)    
            
                    power_2 = tfr_morlet(preproc_S2[condition], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=decim, n_jobs=n_jobs, average = False)                    

                    signal_1 = power_1.data
                    signal_2 = power_2.data
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

                n_epochs = X.shape[0]
                n_channels = X.shape[1]
                n_frequency_bands = X.shape[2]
                n_times = X.shape[3]

                # Initialize an array to hold the PPC values for each epoch, channel, and frequency band
                ppc = np.zeros((n_epochs, n_channels, n_frequency_bands, n_times))

                for epoch in range(n_epochs):
                    for channel in range(n_channels):
                        for frequency_band in range(n_frequency_bands):
                            # Extract the spectral coefficients for the current epoch, channel, and frequency band
                            X_coeff = X[epoch, channel, frequency_band, :]
                            Y_coeff = Y[epoch, channel, frequency_band, :]
                            

                            X_coeff = hilbert(X_coeff)
                            Y_coeff  = hilbert(Y_coeff)

                  
                            X_phase = np.angle(X_coeff)
                            Y_phase = np.angle(Y_coeff)

                            
                            

                            # Compute the instantaneous phase difference
                            phase_difference = np.abs(Y_phase - X_phase) 
                                          
                            # Store the PPC value for the current epoch, channel, and frequency band
                            ppc[epoch, channel, frequency_band,:] = phase_difference

                # now average over the epochs
                ppc_values_averaged = np.mean(ppc, axis = 0)


                pair_ppc_values[condition] = ppc_values_averaged 


            storepath = os.path.join(connectivity_path, 'homologous_tfr_pair_'+ str(pair))
            with open(storepath, "wb") as output_file: 
                pickle.dump(pair_ppc_values, output_file, protocol=pickle.HIGHEST_PROTOCOL)
