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
"""
from hypyp import (
    prep,)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import prep
from hypyp import stats
from hypyp import viz
"""
connectivity_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\time locked connectivity"



n_permutations = 500
# Define the number of contrasts and frequency bands for your grid layout
num_contrasts = 4
num_freq_bands = 4  # Example: Theta, Alpha, Beta, Gamma
# get the montage that we will use
biosemi64_montage = mne.channels.make_standard_montage('biosemi64')

ch_names = biosemi64_montage.ch_names
sfreq = 512  # Example sampling frequency
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
info.set_montage(biosemi64_montage)



# loop through all data files
baseline_stack = np.zeros((64, 44))
individual_stack = np.zeros((64, 44))
sync_egal_stack = np.zeros((64, 44))
leader_stack = np.zeros((64, 44))
follower_stack = np.zeros((64, 44))
complementary_sync_egal_stack = np.zeros((64, 44))
complementary_leader_stack = np.zeros((64, 44))
complementary_follower_stack = np.zeros((64, 44))

stack_dict = dict()
condition_names = ['Synchronous/Egalitarian', 'Synchronous/Leader', 'Synchronous/Follower', 'Complementary/Egalitarian', 'Complementary/Leader', 'Complementary/Follower', 'Start','Individual']
for condition in condition_names:
    condition_stack = np.zeros((64, 44))    
    stack_dict[condition] = condition_stack

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

           # if pair < 15:
            #    continue

            with open(file_path,"rb") as input_file:
                participant_1_power_values, participant_2_power_values = pickle.load(input_file)

         #   if len(participant_1_power_values.keys()) != 7:
         #      continue


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
                print(pair)

                participant_power_values.keys()
                
                #if len(participant_power_values.keys()) != 7:
                 #   continue

                skip = False
                for condition in condition_names:
                        if condition not in participant_power_values.keys():
                            skip = True
                if skip:
                    continue
                
  

                try:
                    

                    for condition in condition_names:

                            condition_data = participant_power_values[condition]
                            
                            # average over time
                            channel_data = np.mean(condition_data[:,:,126:-126], axis = 2)
                            

                            stack_dict[condition] = np.dstack((stack_dict[condition], channel_data))
                except:
                    continue
          
              

# Define the number of contrasts and frequency bands for your grid layout
num_contrasts = 4
num_freq_bands = 4  # Example: Theta, Alpha, Beta, Gamma
# get the montage that we will use
biosemi64_montage = mne.channels.make_standard_montage('biosemi64')

ch_names = biosemi64_montage.ch_names
sfreq = 512  # Example sampling frequency
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
info.set_montage(biosemi64_montage)


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
print(np.shape(stack_dict['Individual']))
baseline_stack = remove_outliers(stack_dict['Start'])
individual_stack = remove_outliers(stack_dict['Individual'])
sync_egal_stack = remove_outliers(stack_dict['Synchronous/Egalitarian'])
leader_stack = remove_outliers(stack_dict['Synchronous/Leader'])
follower_stack = remove_outliers(stack_dict['Synchronous/Follower'])
complementary_sync_egal_stack = remove_outliers(stack_dict['Complementary/Egalitarian'])
complementary_leader_stack = remove_outliers(stack_dict['Complementary/Leader'])
complementary_follower_stack = remove_outliers(stack_dict['Complementary/Follower'])






######################
## NOW DO THE CLUSTER PERMUTATION TEST
######################
adjacency,_ = mne.channels.find_ch_adjacency(info, 'eeg')



import matplotlib.gridspec as gridspec

# Create a figure with a grid of subplots
fig, axs = plt.subplots(num_contrasts, num_freq_bands, figsize=(20, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
# Iterate over each contrast and frequency band to create and plot Evoked objects
fig = plt.figure(figsize=(20, 15))
main_gs = gridspec.GridSpec(num_contrasts, len(freq_bands), figure=fig, wspace = 0.1, hspace = 0.1)
inner_gridspec_dict = {}
for contrast_idx in range(num_contrasts):
   
    for i, freq_band in enumerate(freq_bands.values()):
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=main_gs[contrast_idx, i], height_ratios = [0.6, 1], width_ratios= [1, 1], wspace = 0.5, hspace = 0.8)
        
        
        # Larger subplot (spanning top three rows)
        ax_main = fig.add_subplot(inner_gs[:, :])
        
        # Smaller subplot (bottom left, spanning one row and two columns)
        ax_bottom_left = fig.add_subplot(inner_gs[1, 0])
        
        # Smaller subplot (bottom right, spanning one row and two columns)
        
        inner_gridspec_dict[(contrast_idx, i)] = inner_gs

        #for i, freq_band in enumerate(freq_bands.values()):

         #           X = signal_1[epoch, channel, freq_band[0]:freq_band[1], :]
          #          Y = signal_2[epoch, channel, freq_band[0]:freq_band[1], :]



        # full baseline contrast 
        if contrast_idx == 0:
            data_1 = sync_egal_stack
            data_2 = baseline_stack #individual_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
        
        elif contrast_idx == 1:
            data_1 = leader_stack
            data_2 = baseline_stack #individual_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
          
        elif contrast_idx == 2:
            data_1 = complementary_sync_egal_stack
            data_2 = baseline_stack #individual_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
           
        elif contrast_idx == 3:
            data_1 = complementary_leader_stack
            data_2 = baseline_stack #individual_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
           

        test = 'f oneway'
        factor_levels = 2
     
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
                                                                                        adjacency= adjacency,
                                                                                        t_power=1,
                                                                                        threshold = 2,
                                                                                        out_type='mask')
        significance_level = 0.05
        significant_clusters = cluster_p_values < significance_level

        # Define significance level
        #significance_level = 0.10

        significance_level_05 = 0.05
        significance_level_01 = 0.01

        # Mask for p < 0.05
        mask_05 = np.zeros((64,), dtype=bool)
        # Mask for p < 0.01
        mask_01 = np.zeros((64,), dtype=bool)

        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < significance_level_05:
                cluster_indices = cluster
                mask_05[cluster_indices] = True
            if p_val < significance_level_01:
                cluster_indices = cluster
                mask_01[cluster_indices] = True

        # Define mask_params
        mask_params = dict(marker='o', markerfacecolor='none', markeredgecolor='k', linewidth=0, markersize=4)
        mask_01_params = dict(marker='o', markerfacecolor='white', markeredgecolor='k', linewidth=0, markersize=4)

        
        # Now you have a mask where True indicates channels part of significant clusters

        # Plotting the significant channels on a topomap
        # You might want to create a dummy data array for plotting purposes, since plot_topomap needs data values.
        # One simple approach is to use the mask itself as data, which will highlight significant areas.
        dummy_data = mask_05.astype(float)  # Convert boolean mask to float for plotting

        # Plot the topomap with significant areas highlighted
        # Plot the topomap with significant areas highlighted
        Stat_obs[~mask_05] = 0
        mne.viz.plot_topomap(Stat_obs, info, mask=mask_01, axes=ax_main, sensors=False, show=False, contours = 0, mask_params=mask_01_params)
       # ax_main.set_title(f'Contrast {contrast_idx + 1}, Freq Band {i + 1}\nP-values: {cluster_p_values}')

        #ax_main.set_title(str(cluster_p_values))


        # FULL CONDITION EFFECT
        if contrast_idx == 0:
            data_1 = sync_egal_stack
            data_2 = individual_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
        
        elif contrast_idx == 1:
            data_1 = leader_stack
            data_2 = follower_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
          
        elif contrast_idx == 2:
            data_1 = complementary_sync_egal_stack
            data_2 = sync_egal_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
           
        elif contrast_idx == 3:
            data_1 = complementary_leader_stack
            data_2 = complementary_follower_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
           

        test = 'f oneway'
        factor_levels = 2
     
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
                                                                                        adjacency= adjacency,
                                                                                        t_power=1,
                                                                                        threshold = 2,
                                                                                        out_type='mask')
        significance_level = 0.05
        significant_clusters = cluster_p_values < significance_level

        # Define significance level
        #significance_level = 0.10

        significance_level_05 = 0.05
        significance_level_01 = 0.01

        # Mask for p < 0.05
        mask_05 = np.zeros((64,), dtype=bool)
        # Mask for p < 0.01
        mask_01 = np.zeros((64,), dtype=bool)

        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < significance_level_05:
                cluster_indices = cluster
                mask_05[cluster_indices] = True
            if p_val < significance_level_01:
                cluster_indices = cluster
                mask_01[cluster_indices] = True

        # Define mask_params
        mask_01_params = dict(marker='o', markerfacecolor='white', markeredgecolor='k', linewidth=0, markersize=2)

        
        # Now you have a mask where True indicates channels part of significant clusters

        # Plotting the significant channels on a topomap
        # You might want to create a dummy data array for plotting purposes, since plot_topomap needs data values.
        # One simple approach is to use the mask itself as data, which will highlight significant areas.
        dummy_data = mask_05.astype(float)  # Convert boolean mask to float for plotting

        # Plot the topomap with significant areas highlighted
        # Plot the topomap with significant areas highlighted
        Stat_obs[~mask_05] = 0
          
        # Plot the topomap with significant areas highlighted
        mne.viz.plot_topomap(Stat_obs, info, mask=mask_01, axes=ax_bottom_left, sensors=False, show=False, contours = 0, mask_params=mask_01_params)


        # Show the complete figure
#plt.show()


        #Stat_obs, clusters, cluster_p_values, H0 = stats.statscluster(data = DATA, test = 'f oneway', factor_level = 2, tail = 0, ch_con_freq = scipy.sparse.csr_matrix(metaconn_freq), n_permutations = 100, alpha = 0.05)










################
#Now test contrast between groups
################
















stack_dict = dict()
condition_names = ['Synchronous/Egalitarian', 'Synchronous/Leader', 'Synchronous/Follower', 'Individual', 'Complementary/Egalitarian', 'Complementary/Leader', 'Complementary/Follower']
for contrast in range(4):
    condition_stack = [np.zeros((64, 44)), np.zeros((64, 44))]    
    stack_dict[contrast] = condition_stack



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

           # if pair > 20:
            #    continue

            


            with open(file_path,"rb") as input_file:
                participant_1_power_values, participant_2_power_values = pickle.load(input_file)


            #print(participant_1_power_values)
            print(participant_2_power_values.keys())

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
                print(pair)


                #if len(participant_power_values.keys()) != 7:
                 #   continue

                skip = False
                for condition in condition_names:
                        if condition not in participant_power_values.keys():
                            skip = True
                if skip:
                    continue
                
                # get the contrasted (aka baselined values)
                for contrast in range(4):
                        try: 
                            if contrast == 0:
                                condition_data_1 = participant_power_values['Synchronous/Egalitarian']
                                condition_data_2 = participant_power_values['Start']
                            elif contrast == 1:
                                condition_data_1 = participant_power_values['Synchronous/Leader']
                                condition_data_2 = participant_power_values['Start']
                            elif contrast == 2:
                                condition_data_1 = participant_power_values['Complementary/Egalitarian']
                                condition_data_2 = participant_power_values['Start']
                            elif contrast == 3:
                                condition_data_1 = participant_power_values['Complementary/Leader']
                                condition_data_2 = participant_power_values['Start']
                            
                            # average over time
                            condition_data_1 = np.mean(condition_data_1, axis = 2)
                            condition_data_2 = np.mean(condition_data_2, axis = 2)

                            condition_difference = condition_data_1 - condition_data_2

                            if pair < 21:
                                stack_dict[contrast][0] = np.dstack((stack_dict[contrast][0], condition_difference))
                            else:
                                stack_dict[contrast][1] = np.dstack((stack_dict[contrast][1], condition_difference))
                        except:
                            continue


          
adjacency,_ = mne.channels.find_ch_adjacency(info, 'eeg')

# Create a figure with a grid of subplots
#fig, axs = plt.subplots(num_contrasts, num_freq_bands, figsize=(20, 15))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
# Iterate over each contrast and frequency band to create and plot Evoked objects


for contrast_idx in range(num_contrasts):
   
    for i, freq_band in enumerate(freq_bands.values()):


        # Select the correct axes for the current subplot
        #ax = axs[contrast_idx, i]
           
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=main_gs[contrast_idx, i], height_ratios = [0.7, 1], width_ratios= [1, 1], wspace = 0.5, hspace = 0.8)
        row, col = 1, 2  # Example row and column indices
        inner_gs = inner_gridspec_dict.get((contrast_idx, i))
        
        # Larger subplot (spanning top three rows)
        #ax_main = fig.add_subplot(inner_gs[:, :])
        
        # Smaller subplot (bottom left, spanning one row and two columns)
        #ax_bottom_left = fig.add_subplot(inner_gs[1, 0])
        
        # Smaller subplot (bottom right, spanning one row and two columns)
        ax_bottom_right = fig.add_subplot(inner_gs[1, 1])
    
    
        #for i, freq_band in enumerate(freq_bands.values()):

         #           X = signal_1[epoch, channel, freq_band[0]:freq_band[1], :]
          #          Y = signal_2[epoch, channel, freq_band[0]:freq_band[1], :]

      
        data_1 = stack_dict[contrast_idx][0]
        data_2 = stack_dict[contrast_idx][1]

        print(np.shape(data_1))

        data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]

        test = 'f oneway'
        factor_levels = 2
     
        #metaconn_freq = np.tile(metaconn, (4,4))
        #ch_con_freq = scipy.sparse.csr_matrix(metaconn_freq)

        tail = 0
        alpha = 0.05
        #def stat_fun(*arg):
        # return(scipy.stats.f_oneway(arg[0], arg[1])[0])

        def stat_fun(*arg):
                    return(scipy.stats.ttest_ind(arg[0], arg[1], nan_policy = 'omit')[0])

        Stat_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(data,
                                                                                        stat_fun=stat_fun,
                                                                                        tail=tail,
                                                                                        n_permutations=n_permutations,
                                                                                        adjacency= adjacency,
                                                                                        t_power=1,
                                                                                        threshold = 2,
                                                                                        out_type='mask')
        significance_level = 0.05
        significant_clusters = cluster_p_values < significance_level

        # Define significance level
        #significance_level = 0.10

        significance_level_05 = 0.05
        significance_level_01 = 0.01

        # Mask for p < 0.05
        mask_05 = np.zeros((64,), dtype=bool)
        # Mask for p < 0.01
        mask_01 = np.zeros((64,), dtype=bool)

        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < significance_level_05:
                cluster_indices = cluster
                mask_05[cluster_indices] = True
            if p_val < significance_level_01:
                cluster_indices = cluster
                mask_01[cluster_indices] = True

        # Define mask_params
        mask_01_params = dict(marker='o', markerfacecolor='white', markeredgecolor='k', linewidth=0, markersize=2)

        
        # Now you have a mask where True indicates channels part of significant clusters

        # Plotting the significant channels on a topomap
        # You might want to create a dummy data array for plotting purposes, since plot_topomap needs data values.
        # One simple approach is to use the mask itself as data, which will highlight significant areas.
        dummy_data = mask_05.astype(float)  # Convert boolean mask to float for plotting

        # Plot the topomap with significant areas highlighted
        # Plot the topomap with significant areas highlighted
        Stat_obs[~mask_05] = 0
          
        # Plot the topomap with significant areas highlighted
        mne.viz.plot_topomap(Stat_obs, info, mask=mask_01, axes=ax_bottom_right, sensors=False, show=False, contours = 0, mask_params=mask_01_params)
 

        # Show the complete figure
plt.show()






