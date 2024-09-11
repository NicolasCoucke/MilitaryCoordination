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
import pandas as pd

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

# Step 1: Load the Excel file

excel_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\match_excel_BU_BU_cleaned.xlsx"
 # Update this with the correct path to your file
df = pd.read_excel(excel_path)

# Step 2: Function to filter DataFrame and extract epochs based on condition
def extract_epochs_for_condition(df, pair, condition_string, condition_epochs, sfreq):
    # Filter rows for the specific pair and condition
    filtered_df = df[(df['Pair'] == pair) & (df['Condition'].str.contains(condition_string))]

    # Convert EEG_Begin_Time and EEG_End_Time from string to float (if needed)
    filtered_df['EEG_Begin_Time'] = filtered_df['EEG_Begin_Time'].astype(float)
    filtered_df['EEG_End_Time'] = filtered_df['EEG_End_Time'].astype(float)

    # Initialize a list to store selected epochs
    selected_epochs = []

    # Step 3: Loop through each trial and extract the epochs within EEG time range
    for _, row in filtered_df.iterrows():
        eeg_begin_time = row['EEG_Begin_Time']
        eeg_end_time = row['EEG_End_Time']

        # Convert EEG time to sample indices
        begin_sample = int(eeg_begin_time * sfreq)
        end_sample = int(eeg_end_time * sfreq)

        # Filter epochs based on the event time falling within the specified range
        epoch_indices = np.where((condition_epochs.events[:, 0] >= begin_sample) & (condition_epochs.events[:, 0] <= end_sample))[0]

        # Extract the epochs
        selected_epochs.append(condition_epochs[epoch_indices])

    # Combine all selected epochs
    if selected_epochs:
        combined_epochs = mne.concatenate_epochs(selected_epochs)
    else:
        combined_epochs = None

    return combined_epochs


def get_channel_numbers(channel_names):
    channel_numbers = channel_names
    biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
    # change the channel names in our epochs so that they are the same as the montage
    channels = biosemi64_montage.ch_names
    for i in range(len(channel_numbers)):
        ch_name = channel_names[i]
        index = channels.index(ch_name)
        channel_numbers[i] = index

    return channel_numbers
def calculate_connectivity(pair):
# loop through all data files

    for root, dirs, files in os.walk(prep_path):
        for name in files:
            if 'manual_checked_pair' in name:
               
                file_path = os.path.join(prep_path, name)
                # define paths
                
                split_name = name.split("pair_")
                _pair = int(split_name[1])

                if pair != _pair:
                    continue

                if pair in bad_pairs:
                    continue

                with open(file_path,"rb") as input_file:
                    cleaned_epochs_AR = pickle.load(input_file)

            
                print("processing file " + name)

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
                filtered_event_id = {key: value for key, value in event_id.items() if ((('Synchronous' in key) and ('Success/Checkpoint' in key)) or (('Individual' in key) and ('Success/Checkpoint' in key)) or ('Success/Start' in key) or ('Success/Desyncpoint' in key)) }
                
                # Print the filtered event IDs
                #for key in filtered_event_id.keys():
                 #   print(key)

                preproc_S1 = preproc_S1[list(filtered_event_id.keys())]
                preproc_S2 = preproc_S2[list(filtered_event_id.keys())]

                
                # Define frequencies of interest
                freqs = np.arange(1, 13, 1)  # 1 to 40 Hz in 1 Hz steps

                # just to test do with less freqs instead of 45

                n_cycles = freqs / 4.  # Different number of cycle per frequency

                # Define wavelet parameters
                decim = 1  # To reduce computation time, you can increase this number
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

                participant_1_connectivity_values = dict()
                participant_2_connectivity_values = dict()
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
                    print(condition)

                    try:
                        # now we need to find all the the trials that belong to the pair and the condition
                        condition_epochs_1 = power_1[condition]
                        condition_epochs_2 = power_2[condition]
                    except:
                        continue
                    sfreq = condition_epochs_1.info['sfreq']  # Sampling frequency
                    
                    filtered_df = df[(df['Pair'] == pair) & (df['Condition'].apply(lambda x: x in condition))]

                    # Convert EEG_Begin_Time and EEG_End_Time from string to float (if needed)
                    filtered_df['EEG_Begin_Time'] = filtered_df['EEG_Begin_Time'].astype(float)
                    filtered_df['EEG_End_Time'] = filtered_df['EEG_End_Time'].astype(float)
                    # Initialize a list to store selected epochs
                    
                    #print(filtered_df)
                    #print(filtered_df['Trial'].unique())
                    # Loop through each unique 'Trial' in the filtered DataFrame
                    for trial in filtered_df['Trial'].unique():

                        try:
                            # Get the corresponding row for this trial
                            trial_row = filtered_df[filtered_df['Trial'] == trial]
                            
                            # Extract the 'EEG_Begin_Time' and 'EEG_End_Time' for this trial
                            eeg_begin_time = trial_row['EEG_Begin_Time'].values[0]
                            eeg_end_time = trial_row['EEG_End_Time'].values[0]
                            
                            # Find epochs that fall within the specified time range
                            # Note: epochs.events[:, 0] provides the sample index for each epoch start.
                            # Convert EEG_Begin_Time and EEG_End_Time to sample indices
                            sample_freq = 2048#condition_epochs_1.info['sfreq']
                            start_sample = int(eeg_begin_time * sample_freq) - 10
                            end_sample = int(eeg_end_time * sample_freq)

                           # print(start_sample)
                            #print(condition_epochs_1.events[:, 0])

                            # Select epochs that fall within the specified sample range
                            epochs_in_trial_1 = condition_epochs_1[(condition_epochs_1.events[:, 0] >= start_sample) & (condition_epochs_1.events[:, 0] <= end_sample)]
                            epochs_in_trial_2 = condition_epochs_2[(condition_epochs_2.events[:, 0] >= start_sample) & (condition_epochs_2.events[:, 0] <= end_sample)]
                            
                            

                        
                            # average over time and over epochs within one trial but detect the trials.
                    
                           # print(np.shape(epochs_in_trial_2.data))
                            if 'Start' in condition:
                                average_powers_1 = np.mean(np.abs(epochs_in_trial_1.data[:,:,:,126:512])**2, axis = (0, 3))

                                average_powers_2 = np.mean(np.abs(epochs_in_trial_2.data[:,:,:,126:512])**2, axis = (0, 3))
                            else:
                                average_powers_1 = np.mean(np.abs(epochs_in_trial_1.data[:,:,:,126:-126])**2, axis = (0, 3))

                                average_powers_2 = np.mean(np.abs(epochs_in_trial_2.data[:,:,:,126:-126])**2, axis = (0, 3))

                            participant_1_power_values[condition + '/' + str(trial)] = average_powers_1
                            participant_2_power_values[condition + '/' + str(trial)] = average_powers_2




                            # and now save it for the connectivity

                            signal_1 = epochs_in_trial_1.data
                            signal_2 = epochs_in_trial_2.data
                            #print(np.shape(signal_1))
                            #print(np.shape(signal_2))

                                    # split up into frequencies and calculate for each frequency seperately
                            result_list = []
                            n_epochs, n_ch, n_times =  signal_1.shape[0],  signal_1.shape[1],  signal_1.shape[3]

                            for participant in range(2):

                                if participant ==0:
                                    X = signal_1
                                    Y = signal_2
                                else:
                                    X = signal_2
                                    Y = signal_1

                                n_epochs = signal_1.shape[0]
                                n_channels = signal_1.shape[1]
                                n_frequency_bands = signal_1.shape[2]
                                location_bins = 2

                                right_parietal_channels = ['CP6', 'P8', 'P6', 'P4', 'CP4', 'TP8']
                                right_parietal_channel_numbers = get_channel_numbers(right_parietal_channels)

                                motor_channels = ["C3", "C1", "Cz", "C2", "C4"]
                                motor_channel_numbers = get_channel_numbers( motor_channels)

                                ppc = np.zeros((n_epochs, n_channels, n_frequency_bands, location_bins))

                                locations = [right_parietal_channel_numbers, motor_channel_numbers]
                                for location in range(2):
                                    for epoch in range(n_epochs):
                                        for channel in range(n_channels):
                                            for i in range(n_frequency_bands):
                                                
                                                X = signal_1[epoch, channel, i, 126:-126]

                                                Y = np.nanmean(signal_2[epoch, locations[location], i, 126:-126], axis = 0)

                        

                                                #PROJECTED POWER CORRELATIONS 
                                                X_coeff = X
                                                Y_coeff = Y
                                                
                                                # extract the parts of X and Y that are orthogonal to each other
                                                
                                                X_orthogonal = np.imag((X_coeff) * np.conj(Y_coeff) / np.abs(Y_coeff))
                                                Y_orthogonal = np.imag((Y_coeff) * np.conj(X_coeff) / np.abs(X_coeff))    

                                                # get the absolute values of the orthogonal parts
                                                
                                                X_orthogonal = np.abs(X_orthogonal)
                                                Y_orthogonal = np.abs(Y_orthogonal)


                                                                                

                                                # Compute correlation between the magnitudes of the orthogonal projections
                                                if np.std(Y_orthogonal) * np.std(X_orthogonal) != 0:  # Avoid division by zero
                                                    ppc_value = np.corrcoef(Y_orthogonal, X_orthogonal)[0, 1]
                                                else:
                                                    ppc_value = 0  # Assign a default value in case of std deviation being zero


                                                # compute the PLV between the enveloppes
                                            
                                                ppc[epoch, channel, i, location] = ppc_value

                                                        # (2, n_epochs, n_channels, n_freq_bands, n_times)


                                if participant == 0:
                                    participant_1_connectivity_values[condition + '/' + str(trial)] = ppc
                                else:
                                    participant_2_connectivity_values[condition + '/' + str(trial)] = ppc
                        
                        except KeyError as e:
                            print(f"KeyError: {e} for condition {condition} and trial {trial}")
                            continue
                        except Exception as e:
                            print(f"Unexpected error occurred: {e}")
                            continue

                       # print(np.shape(signal_1))
                       # print(np.shape(signal_2))

                                            
                    
                
            
                storepath = os.path.join(connectivity_path, 'trail_based_individual_tfr_pair'+ str(pair))
                with open(storepath, "wb") as output_file: 
                    pickle.dump([participant_1_power_values, participant_2_power_values], output_file, protocol=pickle.HIGHEST_PROTOCOL)                    
                                


                storepath = os.path.join(connectivity_path, 'trail_based_connectivity_tfr_pair_'+ str(pair))
                with open(storepath, "wb") as output_file: 
                    pickle.dump([participant_1_connectivity_values, participant_2_connectivity_values], output_file, protocol=pickle.HIGHEST_PROTOCOL)


#for pair in range(18,45):#[26, 28, 31, 32, 36, 33, 37, 39, 40, 41, 42, 43, 44]:
#    calculate_connectivity(pair)

for pair in range(1,45):#[26, 28, 31, 32, 36, 33, 37, 39, 40, 41, 42, 43, 44]:
    calculate_connectivity(pair)
"""
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
"""


# Initialize an empty matrix to hold all windows
"""
new WAY FOR TIME DATA STUFF
sub_phase_diff_windows = np.zeros((num_points, window_size))

# Fill the matrix with sliding windows of phase_diff
for i in range(num_points):
    sub_phase_diff_windows[i, :] = phase_diff[i:i+window_size]

# Compute PLV for each window
exp_values = np.exp(1j * sub_phase_diff_windows)
PLV_values = np.abs(np.mean(exp_values, axis=1))
"""

#sub_X_windows = np.zeros((num_points, window_size))
#sub_Y_windows = np.zeros((num_points, window_size))

"""
# Fill the matrix with sliding windows of phase_diff
# Calculate standard deviations
std_Y = np.std(Y_windows, axis=1)
std_X = np.std(X_windows, axis=1)

# Avoid division by zero by setting a mask
non_zero_mask = (std_Y * std_X) != 0

# Initialize time series arrays
ppc_time_series = np.zeros((num_points,))
plv_time_series = np.zeros((num_points,))

# Compute PPC and PLV values for non-zero standard deviation windows
ppc_time_series[non_zero_mask] = np.array([
    np.corrcoef(Y_windows[i], X_windows[i])[0, 1]
    for i in np.where(non_zero_mask)[0]
])

plv_time_series[non_zero_mask] = np.abs(np.sum(np.exp(1j * phase_windows[non_zero_mask]), axis=1) / window_size)

"""
