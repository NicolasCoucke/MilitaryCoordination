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
condition_names = ['Synchronous/Egalitarian', 'Synchronous/LeaderFollower', 'Synchronous/FollowerLeader', 'Individual/Egalitarian', 'Complementary/Egalitarian', 'Complementary/LeaderFollower', 'Complementary/FollowerLeader']
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
        if 'homologous_tfr_pair' in name:
            print("processing file " + name)

            file_path = os.path.join(connectivity_path, name)
            # define paths
            print(file_path)
            
            split_name = name.split("pair_")
            pair = int(split_name[1])

           # if pair < 15:
            #    continue

            


            with open(file_path,"rb") as input_file:
                pair_ppc_values = pickle.load(input_file)


            #if len(participant_1_power_values.keys()) != 7:
            #    continue

        
            #print(participant_power_values )
            #print(np.shape(participant_1_power_values['Synchronous/Egalitarian']))

            # define frequency bands 
            freq_bands = {'Theta': [4, 7],
                            'Alpha': [8, 12],
                            'Beta': [13, 30],
                            'Gamma': [30, 45],
                            'Beta_narrow': [18, 22]}
            freq_bands = OrderedDict(freq_bands)
            # select condition and frequency band
            event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}

            skip = False
            for condition in condition_names:
                    
                    cond_name = str(condition + '/Success/Checkpoint')
                    print(cond_name)
                    if cond_name not in pair_ppc_values.keys():
                        print('skip')
                        skip = True
            if skip:
                continue

        
                

            for condition in condition_names:

                    cond_name = str(condition + '/Success/Checkpoint')
                    condition_data = pair_ppc_values[cond_name]
                    
                    channel_data = np.mean(condition_data[motor_channel_numbers,:,:], axis = 0)
             

                    stack_dict[condition] = np.dstack((stack_dict[condition], channel_data))
    



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
individual_stack = remove_outliers(stack_dict['Individual/Egalitarian'])
sync_egal_stack = remove_outliers(stack_dict['Synchronous/Egalitarian'])
leader_stack = remove_outliers(stack_dict['Synchronous/LeaderFollower'])
follower_stack = remove_outliers(stack_dict['Synchronous/FollowerLeader'])
complementary_sync_egal_stack = remove_outliers(stack_dict['Complementary/Egalitarian'])
complementary_leader_stack = remove_outliers(stack_dict['Complementary/LeaderFollower'])
complementary_follower_stack = remove_outliers(stack_dict['Complementary/FollowerLeader'])




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
            data_2 = sync_egal_stack
            data = [data_1.T, data_2.T]
          
        elif contrast_idx == 2:
            data_1 = complementary_sync_egal_stack
            data_2 = sync_egal_stack
            data = [data_1.T, data_2.T]
           
        elif contrast_idx == 3:
            data_1 = complementary_leader_stack
            data_2 = complementary_sync_egal_stack
            data = [data_1.T, data_2.T]
           
        print(np.shape(data_1.T))
        test = 'f oneway'
        factor_levels = 2
     
        n_permutations = 1000
        #metaconn_freq = np.tile(metaconn, (4,4))
        #ch_con_freq = scipy.sparse.csr_matrix(metaconn_freq)

        tail = 0
        alpha = 0.05
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
frontal_channels = []




power = tfr_morlet(preproc_S1, freqs=freqs, n_cycles=n_cycles, use_fft=True, picks = ['Cz'],
                    return_itc=False, decim=decim, n_jobs=n_jobs, average = True)    


print(np.shape(stack_dict['Synchronous/Egalitarian']))

sync_egal_averaged = np.mean(stack_dict['Synchronous/LeaderFollower'], axis = 2)
print(np.shape(sync_egal_averaged))
power.data  = sync_egal_averaged[np.newaxis,:,:]
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