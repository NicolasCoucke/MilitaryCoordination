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
from my_utils import compute_freq_bands, compute_sync
from scipy.stats import ttest_rel
from mne.time_frequency import tfr_morlet
"""
from hypyp import (
    prep,)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import prep
from hypyp import stats
from hypyp import viz
"""
path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1"

connectivity_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\time locked connectivity"
# loop through all data files
individual_stack = np.zeros((44,512))
sync_egal_stack = np.zeros((44,512))
leader_stack = np.zeros((44,512))
follower_stack = np.zeros((44,512))
complementary_sync_egal_stack = np.zeros((44,512))
complementary_leader_stack = np.zeros((44,512))
complementary_follower_stack = np.zeros((44,512))

stack_dict = dict()
condition_names = ['Synchronous/Egalitarian', 'Synchronous/Leader', 'Synchronous/Follower', 'Complementary/Egalitarian', 'Complementary/Leader', 'Complementary/Follower', 'Start'] #'Individual'
for condition in condition_names:
    condition_stack = np.zeros((44,512))    
    stack_dict[condition] = condition_stack

biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
# change the channel names in our epochs so that they are the same as the montage
channels = biosemi64_montage.ch_names
# get motor channel ids:
motor_channels = ["C3", "C1", "Cz", "C2", "C4"]
motor_channel_numbers = motor_channels
for i in range(len(motor_channels)):
    ch_name = motor_channels[i]
    index = channels.index(ch_name)
    motor_channel_numbers[i] = index




pair = 1
for root, dirs, files in os.walk(connectivity_path):
    for name in files:
        if 'individual_tfr_pair' in name:
            print("processing file " + name)

            file_path = os.path.join(connectivity_path, name)
            # define paths
            print(file_path)
            
            split_name = name.split("pair")
            pair = int(split_name[1])

            if pair > 20:
                continue

            


            with open(file_path,"rb") as input_file:
                participant_1_power_values, participant_2_power_values = pickle.load(input_file)

            #if len(participant_1_power_values.keys()) != 7:
            #    continue

            for participant_power_values in [participant_1_power_values, participant_2_power_values]:
                #print(participant_power_values )
                #print(np.shape(participant_1_power_values['Synchronous/Egalitarian']))

                # define frequency bands 
                freq_bands = {'Theta': [4, 7],
                                'Alpha': [8, 12],
                                'Beta': [13, 30],
                                'Gamma': [30, 45]}
                freq_bands = OrderedDict(freq_bands)
                # select condition and frequency band
                event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}
                skip = False
                for condition in condition_names:
                        if condition not in participant_power_values.keys():
                            skip = True
                if skip:
                    continue

                try:
                    

                    for condition in condition_names:
                        
                            condition_data = participant_power_values[condition]
                            
                            channel_data = np.mean(condition_data[motor_channel_numbers,:,:], axis = 0)
                            

                            stack_dict[condition] = np.dstack((stack_dict[condition], channel_data))
                except:
                    continue


def remove_outliers(data_stack, z_score_threshold=3):
    """
    Remove outliers from a 3D stack of data along the third dimension.
    
    :param data_stack: 3D numpy array from which to remove outliers
    :param z_score_threshold: Threshold of Z-scores to consider as an outlier
    :return: 3D numpy array with outliers set to NaN
    """
    # Calculate mean and standard deviation along the third dimension
    mean = np.mean(data_stack, axis=2, keepdims=True)
    std = np.std(data_stack, axis=2, keepdims=True)
    
    # Calculate Z-scores
    z_scores = (data_stack - mean) / std
    
    # Identify outliers
    outliers = np.abs(z_scores) > z_score_threshold
    
    # Replace outliers with NaN
    data_stack_filtered = np.where(outliers, np.nan, data_stack)
    
    return data_stack_filtered

# Apply the function to each stack
individual_stack = remove_outliers(stack_dict['Start'])
sync_egal_stack = remove_outliers(stack_dict['Synchronous/Egalitarian'])
leader_stack = remove_outliers(stack_dict['Synchronous/Leader'])
follower_stack = remove_outliers(stack_dict['Synchronous/Follower'])
complementary_sync_egal_stack = remove_outliers(stack_dict['Complementary/Egalitarian'])
complementary_leader_stack = remove_outliers(stack_dict['Complementary/Leader'])
complementary_follower_stack = remove_outliers(stack_dict['Complementary/Follower'])

#sync_egal_stack = complementary_sync_egal_stack
# baseline per participant
baseline_indices = range(358,461)
baseline_indices_comp = range(103,206)
for i in range(np.size(sync_egal_stack, 2)):
    for freq in range(np.size(sync_egal_stack, 0)):
        sync_egal_stack[freq,:,i] = 10*np.log10(sync_egal_stack[freq,:,i] / np.mean(individual_stack[freq,:,i]))

for i in range(np.size(sync_egal_stack, 2)):
    for freq in range(np.size(sync_egal_stack, 0)):
        complementary_sync_egal_stack[freq,:,i] = 10*np.log10(complementary_sync_egal_stack[freq,:,i] / np.mean(individual_stack[freq,:,i]))

egal_mean = np.nanmean(sync_egal_stack, axis = 2)
complementary_egal_mean = np.nanmean(complementary_sync_egal_stack, axis = 2)
for freq_key, freq_values in freq_bands.items():
    
    print(np.shape(sync_egal_stack))
    print(np.shape(egal_mean))
    plt.plot(np.nanmean(egal_mean[freq_values[0]:freq_values[1], 128:383], axis = 0))


plt.legend(freq_bands.keys())

for freq_key, freq_values in freq_bands.items():
    
    print(np.shape(sync_egal_stack))
    print(np.shape(egal_mean))
    plt.plot(np.nanmean(complementary_egal_mean[freq_values[0]:freq_values[1], 128:383], axis = 0), linestyle = '--')


plt.axvline(x = 103, color = 'black')
plt.axvline(x = 154, linestyle = '--', color = 'black')
plt.show()



sync_egal_averaged = np.nanmean(stack_dict['Synchronous/Egalitarian'], axis = 2)

# Define frequencies of interest
freqs = np.arange(1, 45, 1)  # 1 to 40 Hz in 1 Hz steps
n_cycles = freqs / 4.  # Different number of cycle per frequency

# Define wavelet parameters
decim = 2  # To reduce computation time, you can increase this number
n_jobs = 1  # Number of parallel jobs to run. Can be increased if your machine supports it.

motor_channels = ["C3", "C1", "Cz", "C2", "C4"]


prep_path = os.path.join(path, "time locked preprocessed")
file_path = os.path.join(prep_path, "pair_1")
with open(file_path,"rb") as input_file:
    cleaned_epochs_AR = pickle.load(input_file)

preproc_S1 = cleaned_epochs_AR[0]


preproc_S1 = preproc_S1['Individual']



power = tfr_morlet(preproc_S1, freqs=freqs, n_cycles=n_cycles, use_fft=True, picks = ["C3", "C1", "Cz", "C2", "C4"],
                    return_itc=False, decim=decim, n_jobs=n_jobs, average = True)    

sync_egal_averaged = np.nanmean(stack_dict['Synchronous/Leader']/stack_dict['Synchronous/Follower'], axis = 2)

print()

power.data = sync_egal_averaged[np.newaxis,:,:]


power.plot(baseline=(0.5, 0.7), tmin = -0.5, tmax = 0.5, mode='logratio', title='MEG 0211 Power')




######################
## NOW DO THE CLUSTER PERMUTATION TEST
######################


num_locations = 1
num_contrasts = 4
# Create a figure with a grid of subplots
fig, axs = plt.subplots(4, num_locations, figsize=(20, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
# Iterate over each contrast and frequency band to create and plot Evoked objects
for contrast_idx in range(num_contrasts):
   
    for loc in range(1):


        # Select the correct axes for the current subplot
        ax = axs[contrast_idx]
           
        
    
        #for i, freq_band in enumerate(freq_bands.values()):

         #           X = signal_1[epoch, channel, freq_band[0]:freq_band[1], :]
          #          Y = signal_2[epoch, channel, freq_band[0]:freq_band[1], :]

        if contrast_idx == 0:
            data_1 = sync_egal_stack
            data_2 = individual_stack
            data = [data_1.T, data_2.T]
        
        elif contrast_idx == 1:
            data_1 = leader_stack
            data_2 = follower_stack
            data = [data_1.T, data_2.T]
          
        elif contrast_idx == 2:
            data_1 = complementary_sync_egal_stack
            data_2 = sync_egal_stack
            data = [data_1.T, data_2.T]
           
        elif contrast_idx == 3:
            data_1 = complementary_leader_stack
            data_2 = complementary_follower_stack
            data = [data_1.T, data_2.T]
           
        print(np.shape(data_1.T))
        test = 'f oneway'
        factor_levels = 2
     
        n_permutations = 11
        #metaconn_freq = np.tile(metaconn, (4,4))
        #ch_con_freq = scipy.sparse.csr_matrix(metaconn_freq)

        tail = 0
        alpha = 0.10
        #def stat_fun(*arg):
        # return(scipy.stats.f_oneway(arg[0], arg[1])[0])

        def stat_fun(*arg):
                    return(scipy.stats.ttest_rel(arg[0], arg[1], nan_policy = 'omit')[0])

        Stat_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(data,
                                                                                        stat_fun=stat_fun,
                                                                                        tail=tail,
                                                                                        n_permutations=n_permutations,
                                                                                        t_power=1,
                                                                                        threshold = 2,
                                                                                        out_type='mask')
        significance_level = 0.05
        significant_clusters = cluster_p_values < significance_level

        # Define significance level
        #significance_level = 0.10

        # Initialize a mask with False (indicating no significance)
        mask = np.zeros((44,512), dtype=bool)  # data.shape[1] is n_channels

        # Iterate through clusters and their p-values
        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < significance_level:
                
                # If significant, set those channel indices in the mask to True
                # Assuming clusters is a list of tuples where the first element is the cluster index array
                cluster_indices = cluster.T
                mask[cluster_indices] = True

        # Now you have a mask where True indicates channels part of significant clusters

        # Plotting the significant channels on a topomap
        # You might want to create a dummy data array for plotting purposes, since plot_topomap needs data values.
        # One simple approach is to use the mask itself as data, which will highlight significant areas.
        dummy_data = mask.astype(float)  # Convert boolean mask to float for plotting
        # create_epoch_object
        pair = 1


        prep_path = os.path.join(path, "time locked preprocessed")
        file_path = os.path.join(prep_path, "pair_1")
        with open(file_path,"rb") as input_file:
            cleaned_epochs_AR = pickle.load(input_file)

        preproc_S1 = cleaned_epochs_AR[0]


        preproc_S1 = preproc_S1['Individual']

        # Define frequencies of interest
        freqs = np.arange(1, 45, 1)  # 1 to 40 Hz in 1 Hz steps
        n_cycles = freqs / 4.  # Different number of cycle per frequency

        # Define wavelet parameters
        decim = 2  # To reduce computation time, you can increase this number
        n_jobs = 1  # Number of parallel jobs to run. Can be increased if your machine supports it.

        motor_channels = ["C3", "C1", "Cz", "C2", "C4"]

    
        power = tfr_morlet(preproc_S1, freqs=freqs, n_cycles=n_cycles, use_fft=True, picks = ['Cz'],
                    return_itc=False, decim=decim, n_jobs=n_jobs, average = True)    
        
        power.data = Stat_obs.T
        ax.imshow(Stat_obs.T, aspect = 'auto')


        
        # Create an RGBA color mask
        rgba_mask = np.zeros((*mask.shape, 4))  # Initialize a transparent RGBA mask
        rgba_mask[mask] = [1, 0, 0, 1]  # Set red color with full opacity for True values in the mask
        rgba_mask[~mask] = [0, 0, 0, 0]  # Set fully transparent for False values

        # Overlay the RGBA mask on top of the original plot
        ax.imshow(rgba_mask, aspect='auto', interpolation='none')

        #plt.show()


       # power.plot(baseline= None, tmin = -0.5, tmax = 0.5, mode='logratio', mask = mask, mask_style = 'contour')

        # Show the complete figure
plt.show()



# create_epoch_object
pair = 1


prep_path = os.path.join(path, "time locked preprocessed")
file_path = os.path.join(prep_path, "pair_1")
with open(file_path,"rb") as input_file:
    cleaned_epochs_AR = pickle.load(input_file)

preproc_S1 = cleaned_epochs_AR[0]


preproc_S1 = preproc_S1['Individual']

# Define frequencies of interest
freqs = np.arange(1, 45, 1)  # 1 to 40 Hz in 1 Hz steps
n_cycles = freqs / 4.  # Different number of cycle per frequency

# Define wavelet parameters
decim = 2  # To reduce computation time, you can increase this number
n_jobs = 1  # Number of parallel jobs to run. Can be increased if your machine supports it.

motor_channels = ["C3", "C1", "Cz", "C2", "C4"]





#motor_channels = [ "Cz"]

frontal_channels = []




power_1 = tfr_morlet(preproc_S1, freqs=freqs, n_cycles=n_cycles, use_fft=True, picks = ['Cz'],
                    return_itc=False, decim=decim, n_jobs=n_jobs, average = True)    

power_2 = tfr_morlet(preproc_S1, freqs=freqs, n_cycles=n_cycles, use_fft=True, picks = ['Cz'],
                    return_itc=False, decim=decim, n_jobs=n_jobs, average = True)    



epoch_mean = np.nanmean(stack_dict['Synchronous/Egalitarian'], axis = 2)
freq_mean = np.nanmean(epoch_mean[8:12], axis = 0)
beta_mean = np.nanmean(epoch_mean[28:30], axis = 0)

plt.plot(freq_mean)
plt.plot(beta_mean)

epoch_mean = np.nanmean(stack_dict['Complementary/Egalitarian'], axis = 2)
freq_mean = np.nanmean(epoch_mean[8:12], axis = 0)
beta_mean = np.nanmean(epoch_mean[28:30], axis = 0)
plt.plot(freq_mean)
plt.plot(beta_mean)
plt.show()



power = tfr_morlet(preproc_S1, freqs=freqs, n_cycles=n_cycles, use_fft=True, picks = ['Cz'],
                    return_itc=False, decim=decim, n_jobs=n_jobs, average = True)    




sync_egal_averaged = np.nanmean(stack_dict['Complementary/Egalitarian'] / stack_dict['Synchronous/Egalitarian'], axis = 2)

power.data = sync_egal_averaged[np.newaxis,:,:]


power.plot(baseline=(-0.5, -0.4), tmin = -0.5, tmax = 0.5, mode='logratio', title='MEG 0211 Power')


"""
 print(np.shape(participant_power_values['Individual']))
individual_stack = np.dstack((individual_stack, participant_power_values['Individual']))
sync_egal_stack = np.dstack((sync_egal_stack , participant_power_values['Synchronous/Egalitarian']))
leader_stack = np.dstack((leader_stack , participant_power_values['Synchronous/Leader']))
follower_stack = np.dstack((follower_stack , participant_power_values['Synchronous/Follower']))
complementary_sync_egal_stack = np.dstack((complementary_sync_egal_stack , participant_power_values['Complementary/Egalitarian']))
complementary_leader_stack = np.dstack((complementary_leader_stack , participant_power_values['Complementary/Leader']))
complementary_follower_stack = np.dstack((complementary_follower_stack , participant_power_values['Complementary/Follower']))
"""