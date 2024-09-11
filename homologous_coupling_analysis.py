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
from stat_utils import apply_fisher_z_transform
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
#connectivity_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\time locked connectivity"

stack_dict = dict()
condition_names = ['Synchronous/Egalitarian', 'Synchronous/LeaderFollower', 'Synchronous/FollowerLeader', 'Individual/Egalitarian', 'Complementary/Egalitarian', 'Complementary/LeaderFollower', 'Complementary/FollowerLeader']
for contrast in range(4):
    condition_stack = [np.zeros((64, 44)), np.zeros((64, 44))]    
    stack_dict[contrast] = condition_stack

# Define the number of contrasts and frequency bands for your grid layout
num_contrasts = 4
num_freq_bands = 4  # Example: Theta, Alpha, Beta, Gamma
# get the montage that we will use
biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
ch_names = biosemi64_montage.ch_names
sfreq = 512  # Example sampling frequency
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
info.set_montage(biosemi64_montage)
                 
n_permutations = 500

# define frequency bands 
freq_bands = {'Theta': [4, 7],
                'Alpha': [8, 12],
                'Beta': [13, 30],
                'Gamma': [30, 45]}
freq_bands = OrderedDict(freq_bands)
# select condition and frequency band
event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual/Egalitarian': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}



############
# MAIN EFFECT FOR EACH CONTRAST
############



# loop through all data files
baseline_stack = np.zeros((64,44))
individual_stack = np.zeros((64,44))
sync_egal_stack = np.zeros((64,44))
leader_stack = np.zeros((64,44))
follower_stack = np.zeros((64,44))
complementary_sync_egal_stack = np.zeros((64,44))
complementary_leader_stack = np.zeros((64,44))
complementary_follower_stack = np.zeros((64,44))

stack_dict = dict()
condition_names = ['Synchronous/Egalitarian', 'Synchronous/LeaderFollower', 'Synchronous/FollowerLeader', 'Individual/Egalitarian', 'Complementary/Egalitarian', 'Complementary/LeaderFollower', 'Complementary/FollowerLeader']
for condition in condition_names:
    condition_stack = np.zeros((64,44))    
    stack_dict[condition] = condition_stack



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


import pandas as pd
data = []  # Starting with an empty list, but dictionaries can be appended as new rows
df = pd.DataFrame(data)

pair = 1
for root, dirs, files in os.walk(connectivity_path):
    for name in files:
        if 'homologous_connectivity_pair' in name:

            file_path = os.path.join(connectivity_path, name)
            # define paths
            print(file_path)
            
            split_name = name.split("pair_")
            pair = int(split_name[1])


            #if pair > 20:
             #   continue
 

            
            with open(file_path,"rb") as input_file:
                PLV_values, ppc_values = pickle.load(input_file)

           # print(ppc_values.keys())
            

            values = apply_fisher_z_transform(ppc_values)
            #values = ppc_values
            print(ppc_values.keys())
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
           
            group = 'none'
            if pair > 20:
                group = 'civilian'
            else:
                group = 'military'

            
            #if len(values.keys()) != 14:
            #    continue
        
            try:
                
                start_keys = [key for key in values.keys() if 'Start' in key]
                arrays_to_average = []
                for key in start_keys:
                    arrays_to_average.append(values[key])
                average_baseline_values = np.mean(np.stack(arrays_to_average, axis=-1), axis=-1)
                
                

                individual_values = values['Individual/Egalitarian/Success/Checkpoint']
                  

                sync_egal_values = values['Synchronous/Egalitarian/Success/Checkpoint']  
                

                try:
                    sync_leader_values = values['Synchronous/LeaderFollower/Success/Checkpoint']
                except:
                    sync_leader_values = values['Synchronous/FollowerLeader/Success/Checkpoint']
                

                complementary_sync_egal_values = values['Complementary/Egalitarian/Success/Desyncpoint']
                

                try:
                    complementary_leader_values = values['Complementary/LeaderFollower/Success/Desyncpoint']
                except:
                    complementary_leader_values = values['Complementary/FollowerLeader/Success/Desyncpoint']
                
            except:
                print('pair' + str(pair) + 'not working')
                continue
            

            baseline_stack = np.dstack((baseline_stack, average_baseline_values))
            individual_stack = np.dstack((individual_stack, individual_values))   
            sync_egal_stack = np.dstack((sync_egal_stack, sync_egal_values))
            leader_stack = np.dstack((leader_stack , values['Synchronous/FollowerLeader/Success/Checkpoint']))
            complementary_sync_egal_stack = np.dstack((complementary_sync_egal_stack, complementary_sync_egal_values))
            complementary_leader_stack = np.dstack((complementary_leader_stack , complementary_leader_values))
            condition_dict = dict({'Synchronous/Egalitarian': sync_egal_values, 'Synchronous/Hierarchical': sync_leader_values, 'Complementary/Egalitarian': complementary_sync_egal_values, 'Complementary/Hierarchical': complementary_leader_values, 'Individual': individual_values})
                

            for condition in condition_dict.keys():
                condition_data = condition_dict[condition]
                condition_data = condition_data
                baseline_data = average_baseline_values
                # now baseline the condition data
                #condition_data = condition_data - baseline_data

                #condition_data = 10*np.log(condition_data)

                if 'Individual' == condition:
                    synchrony = 'Alone'
                    hierarchy = 'Egalitarian'
                elif 'Synchronous' in condition:
                    synchrony = 'Synchronous'
                elif 'Complementary' in condition:
                    synchrony = 'Complementary'
                else:
                    continue
                
                leader = 'none'
                if 'Hierarchical' in condition:
                    hierarchy = 'Hierarchical'
                elif 'Egalitarian' in condition:
                    hierarchy = 'Egalitarian'
                else:
                    if condition != 'Individual':
                        continue

                right_parietal_channels = ['CP6', 'P8', 'P6', 'P4', 'CP4', 'TP8']
                motor_channels = ["C3", "C1", "Cz", "C2", "C4"]
                frontal_channels = ["AF3", "AFz", "AF4", "F3", "F1", "Fz", "F2", "F4"]

                channel_numbers = get_channel_numbers(right_parietal_channels)
                alpha_rightparietal = np.nanmean(condition_data[channel_numbers, 8:12])

                channel_numbers = get_channel_numbers(frontal_channels)
                alpha_frontal = np.nanmean(condition_data[channel_numbers, 8:12])

                channel_numbers = get_channel_numbers(motor_channels)
                alpha_central = np.nanmean(condition_data[channel_numbers, 8:12])

                beta_central = np.nanmean(condition_data[channel_numbers, 13:30])


                #channel_data = np.mean(condition_data[motor_channel_numbers,:,:], axis = 0)
                df = df._append({'pair': pair, 'group': group, 'synchrony': synchrony, 'hierarchy': hierarchy, 'leader': leader, 'Alpha_RightParietal': alpha_rightparietal, 'Alpha_Frontal': alpha_central, 'Alpha_Central': alpha_central, 'Beta_Central': beta_central, 'baseline': 'yes'}, ignore_index=True)

storepath = os.path.join(path, 'homologous_ROI_results.csv')
df.to_csv(storepath, index=False)           







def remove_outliers(data_stack, z_score_threshold=2):
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



individual_stack = remove_outliers(individual_stack)
sync_egal_stack = remove_outliers(sync_egal_stack)
leader_stack = remove_outliers(leader_stack)
complementary_sync_egal_stack = remove_outliers(complementary_sync_egal_stack)
complementary_leader_stack = remove_outliers(complementary_leader_stack)




adjacency,_ = mne.channels.find_ch_adjacency(info, 'eeg')


   # define frequency bands 
freq_bands = {'Theta': [4, 7],
                'Alpha': [8, 12],
                'Beta': [13, 30],
                'Gamma': [30, 45]}



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
        
        # FULL CONDITIONEFFECT
        if contrast_idx == 0:
            data = [np.nanmean(sync_egal_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.nanmean(baseline_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T]
        
        elif contrast_idx == 1:
            data = [np.nanmean(leader_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.nanmean(baseline_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T]
          
        elif contrast_idx == 2:
            data = [np.nanmean(complementary_sync_egal_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.nanmean(baseline_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T]
           
        elif contrast_idx == 3:
            data = [np.nanmean(complementary_leader_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.nanmean(baseline_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T]
        
        for i in data:
            print(np.shape(i))
      

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
        print(np.shape(Stat_obs))

        significance_level = 0.05
        significant_clusters = cluster_p_values < significance_level

        # Define significance level
        #significance_level = 0.10

        # Initialize a mask with False (indicating no significance)
        mask = np.zeros((64,), dtype=bool)  # data.shape[1] is n_channels

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


        # CONDITION EFFECT
        if contrast_idx == 0:
            data_1 = sync_egal_stack
            data_2 = individual_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
        
        elif contrast_idx == 1:
            data_1 = leader_stack
            data_2 = sync_egal_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
          
        elif contrast_idx == 2:
            data_1 = complementary_sync_egal_stack
            data_2 = sync_egal_stack
            data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
           
        elif contrast_idx == 3:
            data_1 = complementary_leader_stack
            data_2 = complementary_sync_egal_stack
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







############
# CONTRAST BETWEEN GROUPS
############


stack_dict = dict()
for contrast in range(4):
    condition_stack = [np.zeros((64, 44)), np.zeros((64, 44))]    
    stack_dict[contrast] = condition_stack


pair = 1
for root, dirs, files in os.walk(connectivity_path):
    for name in files:
         if 'homologous_connectivity_pair' in name:

            file_path = os.path.join(connectivity_path, name)
            # define paths
            print(file_path)
            
            split_name = name.split("pair_")
            pair = int(split_name[1])


            
            with open(file_path,"rb") as input_file:
                imcoh_values, ppc_values = pickle.load(input_file)

            values = apply_fisher_z_transform(ppc_values)

            print(values.keys())
            #print(participant_power_values )
            #print(np.shape(participant_1_power_values['Synchronous/Egalitarian']))

            # Apply the function to each stack
            start_keys = [key for key in values.keys() if 'Start' in key]
            arrays_to_average = []
            for key in start_keys:
                arrays_to_average.append(values[key])
            average_baseline_values = np.mean(np.stack(arrays_to_average, axis=-1), axis=-1)



            
            # get the contrasted (aka baselined values)
            for contrast in range(4):
                try:
                    if contrast == 0:
                        condition_data_1 = values['Synchronous/Egalitarian/Success/Checkpoint']
                        condition_data_2 = average_baseline_values #values['Individual/Egalitarian']
                    elif contrast == 1:
                        try:
                            condition_data_1 =  values['Synchronous/LeaderFollower/Success/Checkpoint']
                        except:
                            condition_data_1 =  values['Synchronous/FollowerLeader/Success/Checkpoint']
                        condition_data_2 = average_baseline_values #values['Synchronous/Egalitarian']
                    elif contrast == 2:
                        condition_data_1 = values['Complementary/Egalitarian/Success/Desyncpoint']
                        condition_data_2 = average_baseline_values #values['Synchronous/Egalitarian']
                    elif contrast == 3:
                        try:
                            condition_data_1 =  values['Complementary/LeaderFollower/Success/Desyncpoint']
                        except:
                            condition_data_1 =  values['Complementary/FollowerLeader/Success/Desyncpoint']
                        condition_data_2 = average_baseline_values #values['Complementary/Egalitarian']
                    

                    condition_difference = condition_data_1 - condition_data_2
                    
                    

                    if pair < 21:
                        stack_dict[contrast][0] = np.dstack((stack_dict[contrast][0], condition_difference))
                        print('mil')
                    else:
                        stack_dict[contrast][1] = np.dstack((stack_dict[contrast][1], condition_difference))
                        print('civ')
                except:
                    continue








# Iterate over each contrast and frequency band to create and plot Evoked objects
for contrast_idx in range(num_contrasts):
   
    for i, freq_band in enumerate(freq_bands.values()):


            
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=main_gs[contrast_idx, i], height_ratios = [0.7, 1], width_ratios= [1, 1], wspace = 0.5, hspace = 0.8)
        row, col = 1, 2  # Example row and column indices
        inner_gs = inner_gridspec_dict.get((contrast_idx, i))
        
        # Larger subplot (spanning top three rows)
        #ax_main = fig.add_subplot(inner_gs[:, :])
        
        # Smaller subplot (bottom left, spanning one row and two columns)
        #ax_bottom_left = fig.add_subplot(inner_gs[1, 0])
        
        # Smaller subplot (bottom right, spanning one row and two columns)
        ax_bottom_right = fig.add_subplot(inner_gs[1, 1])
    
        
      
        data_1 = stack_dict[contrast_idx][0]
        data_2 = stack_dict[contrast_idx][1]


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




