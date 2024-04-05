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
connectivity_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\time locked connectivity"


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
                imcoh_values, ppc_values = pickle.load(input_file)

            values = apply_fisher_z_transform(imcoh_values)

            
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
           

            
            if len(values.keys()) != 7:
                continue
        

            try:
                individual_stack = np.dstack((individual_stack, values['Individual/Egalitarian/Success/Checkpoint']))
                sync_egal_stack = np.dstack((sync_egal_stack , values['Synchronous/Egalitarian/Success/Checkpoint']))


                try:
                    leader_stack = np.dstack((leader_stack , values['Synchronous/LeaderFollower/Success/Checkpoint']))
                except:
                    leader_stack = np.dstack((leader_stack , values['Synchronous/FollowerLeader/Success/Checkpoint']))

            
                complementary_sync_egal_stack = np.dstack((complementary_sync_egal_stack , values['Complementary/Egalitarian/Success/Checkpoint']))
            
                try:
                    complementary_leader_stack = np.dstack((complementary_leader_stack , values['Complementary/LeaderFollower/Success/Checkpoint']))
                except:
                    complementary_leader_stack = np.dstack((complementary_follower_stack , values['Complementary/FollowerLeader/Success/Checkpoint']))
            except:
                continue
            print('worked')
           







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
# Create a figure with a grid of subplots
fig, axs = plt.subplots(num_contrasts, num_freq_bands, figsize=(20, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
# Iterate over each contrast and frequency band to create and plot Evoked objects



for contrast_idx in range(num_contrasts):
   
    for i, freq_band in enumerate(freq_bands.values()):


        # Select the correct axes for the current subplot
        ax = axs[contrast_idx, i]
        
      
        #data_1 = stack_dict[contrast_idx][0]
        #data_2 = stack_dict[contrast_idx][1]


       # data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]
        if contrast_idx == 0:
            data = [np.nanmean(sync_egal_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.nanmean(individual_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T]
        
        elif contrast_idx == 1:
            data = [np.nanmean(leader_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.nanmean(individual_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T]
          
        elif contrast_idx == 2:
            data = [np.nanmean(complementary_sync_egal_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.nanmean(individual_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T]
           
        elif contrast_idx == 3:
            data = [np.nanmean(complementary_leader_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.nanmean(individual_stack[:,freq_band[0]:freq_band[1],:], axis = 1).T]
           

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

        # Iterate through clusters and their p-values
        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < significance_level:
                print(cluster)
                # If significant, set those channel indices in the mask to True
                # Assuming clusters is a list of tuples where the first element is the cluster index array
                cluster_indices = cluster
                mask[cluster_indices] = True

        # Now you have a mask where True indicates channels part of significant clusters

        # Plotting the significant channels on a topomap
        # You might want to create a dummy data array for plotting purposes, since plot_topomap needs data values.
        # One simple approach is to use the mask itself as data, which will highlight significant areas.
        dummy_data = mask.astype(float)  # Convert boolean mask to float for plotting

        # Plot the topomap with significant areas highlighted
        mne.viz.plot_topomap(Stat_obs, info, mask=mask, axes=ax, sensors=True, show=False)

 

        # Show the complete figure
plt.show()



















############
# CONTRAST BETWEEN GROUPS
############



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

            values = apply_fisher_z_transform(imcoh_values)

            print(values.keys())
            #print(participant_power_values )
            #print(np.shape(participant_1_power_values['Synchronous/Egalitarian']))




            
            # get the contrasted (aka baselined values)
            for contrast in range(4):
                    try: 
                        if contrast == 0:
                            condition_data_1 = values['Synchronous/Egalitarian']
                            condition_data_2 = values['Individual/Egalitarian']
                        elif contrast == 1:
                            try:
                                condition_data_1 =  values['Synchronous/LeaderFollower']
                            except:
                                condition_data_1 =  values['Synchronous/LeaderFollower']
                            condition_data_2 = values['Synchronous/Egalitarian']
                        elif contrast == 2:
                            condition_data_1 = values['Complementary/Egalitarian']
                            condition_data_2 = values['Synchronous/Egalitarian']
                        elif contrast == 3:
                            try:
                                condition_data_1 =  values['Complementary/LeaderFollower']
                            except:
                                condition_data_1 =  values['Complementary/LeaderFollower']
                            condition_data_2 = values['Complementary/Egalitarian']
                        

                        condition_difference = condition_data_1 - condition_data_2
                        
                       

                        if pair < 21:
                            stack_dict[contrast][0] = np.dstack((stack_dict[contrast][0], condition_difference))
                        else:
                            stack_dict[contrast][1] = np.dstack((stack_dict[contrast][1], condition_difference))
                    except:
                        continue
plt.imshow(condition_data_1, aspect = 'auto')
plt.show()





adjacency,_ = mne.channels.find_ch_adjacency(info, 'eeg')


# Create a figure with a grid of subplots
fig, axs = plt.subplots(num_contrasts, num_freq_bands, figsize=(20, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
# Iterate over each contrast and frequency band to create and plot Evoked objects
for contrast_idx in range(num_contrasts):
   
    for i, freq_band in enumerate(freq_bands.values()):


        # Select the correct axes for the current subplot
        ax = axs[contrast_idx, i]
        
      
        data_1 = stack_dict[contrast_idx][0]
        data_2 = stack_dict[contrast_idx][1]


        data = [np.mean(data_1[:,freq_band[0]:freq_band[1],:], axis = 1).T, np.mean(data_2[:,freq_band[0]:freq_band[1],:], axis = 1).T]

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

        # Initialize a mask with False (indicating no significance)
        mask = np.zeros((64,), dtype=bool)  # data.shape[1] is n_channels

        # Iterate through clusters and their p-values
        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < significance_level:
                print(cluster)
                # If significant, set those channel indices in the mask to True
                # Assuming clusters is a list of tuples where the first element is the cluster index array
                cluster_indices = cluster
                mask[cluster_indices] = True

        # Now you have a mask where True indicates channels part of significant clusters

        # Plotting the significant channels on a topomap
        # You might want to create a dummy data array for plotting purposes, since plot_topomap needs data values.
        # One simple approach is to use the mask itself as data, which will highlight significant areas.
        dummy_data = mask.astype(float)  # Convert boolean mask to float for plotting

        # Plot the topomap with significant areas highlighted
        mne.viz.plot_topomap(Stat_obs, info, mask=mask, axes=ax, sensors=True, show=False)
        ax.set_title(str(cluster_p_values))

 

        # Show the complete figure
plt.show()          



