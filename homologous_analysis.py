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
# loop through all data files
individual_stack = np.zeros((64,5))
sync_egal_stack = np.zeros((64,5))
leader_stack = np.zeros((64,5))
follower_stack = np.zeros((64,5))
complementary_sync_egal_stack = np.zeros((64,5))
complementary_leader_stack = np.zeros((64,5))
complementary_follower_stack = np.zeros((64,5))

pair = 1
for root, dirs, files in os.walk(connectivity_path):
    for name in files:
        if 'homologous_pair' in name:

            file_path = os.path.join(connectivity_path, name)
            # define paths
            print(file_path)
            
            split_name = name.split("pair_")
            pair = int(split_name[1])


 

            
            with open(file_path,"rb") as input_file:
                imcoh_values, ppc_values = pickle.load(input_file)

            values = ppc_values
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
            print(values.keys())

            
            
            individual_stack = np.dstack((individual_stack, values['Individual']))
            #print(np.shape(values['Individual']))
            try:
                individual_stack = np.dstack((individual_stack, values['Individual']))
                sync_egal_stack = np.dstack((sync_egal_stack , values['Synchronous/Egalitarian']))


                try:
                    leader_stack = np.dstack((leader_stack , values['Synchronous/LeaderFollower']))
                except:
                    leader_stack = np.dstack((leader_stack , values['Synchronous/FollowerLeader']))

            
                complementary_sync_egal_stack = np.dstack((complementary_sync_egal_stack , values['Complementary/Egalitarian']))
            
                try:
                    complementary_leader_stack = np.dstack((complementary_leader_stack , values['Complementary/LeaderFollower']))
                except:
                    complementary_leader_stack = np.dstack((complementary_follower_stack , values['Complementary/FollowerLeader']))
            except:
                continue

print(np.shape(individual_stack))
individual_stack = individual_stack[:,:,1:]
sync_egal_stack = sync_egal_stack[:,:,1:]
leader_stack = leader_stack[:,:,1:]
#follower_stack = follower_stack[:,:,1:]
complementary_sync_egal_stack = complementary_sync_egal_stack[:,:,1:]
complementary_leader_stack = complementary_leader_stack[:,:,1:]
#complementary_follower_stack = complementary_follower_stack[:,:,1:]

# Define the number of contrasts and frequency bands for your grid layout
num_contrasts = 4
num_freq_bands = 5  # Example: Theta, Alpha, Beta, Gamma
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
individual_stack = remove_outliers(individual_stack)
sync_egal_stack = remove_outliers(sync_egal_stack)
leader_stack = remove_outliers(leader_stack)
complementary_sync_egal_stack = remove_outliers(complementary_sync_egal_stack)
complementary_leader_stack = remove_outliers(complementary_leader_stack)






######################
## NOW DO THE CLUSTER PERMUTATION TEST
######################
adjacency,_ = mne.channels.find_ch_adjacency(info, 'eeg')



# Create a figure with a grid of subplots
fig, axs = plt.subplots(num_contrasts, num_freq_bands, figsize=(20, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
# Iterate over each contrast and frequency band to create and plot Evoked objects
for contrast_idx in range(num_contrasts):
   
    for freq_band in range(num_freq_bands):


        # Select the correct axes for the current subplot
        ax = axs[contrast_idx, freq_band]
           


        if contrast_idx == 0:
            data = [sync_egal_stack[:,freq_band,:].T, individual_stack[:,freq_band,:].T]
        
        elif contrast_idx == 1:
            data = [leader_stack[:,freq_band,:].T, sync_egal_stack[:,freq_band,:].T]
          
        elif contrast_idx == 2:
            data = [complementary_sync_egal_stack[:,freq_band,:].T, sync_egal_stack[:,freq_band,:].T]
           
        elif contrast_idx == 3:
            data = [complementary_leader_stack[:,freq_band,:].T, complementary_sync_egal_stack[:,freq_band,:].T]
           

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
        mne.viz.plot_topomap(dummy_data, info, mask=mask, cmap='Reds', axes=ax, sensors=True, show=False)

 

        # Show the complete figure
plt.show()


        #Stat_obs, clusters, cluster_p_values, H0 = stats.statscluster(data = DATA, test = 'f oneway', factor_level = 2, tail = 0, ch_con_freq = scipy.sparse.csr_matrix(metaconn_freq), n_permutations = 100, alpha = 0.05)





# Create a figure with a grid of subplots
fig, axs = plt.subplots(num_contrasts, num_freq_bands, figsize=(20, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots

# Iterate over each contrast and frequency band to create and plot Evoked objects
for contrast_idx in range(num_contrasts):
    if contrast_idx == 0:
        t_stats, p_values = ttest_rel(sync_egal_stack, individual_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 1:
        t_stats, p_values = ttest_rel(leader_stack, sync_egal_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 2:
        t_stats, p_values = ttest_rel(complementary_sync_egal_stack, sync_egal_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 3:
        t_stats, p_values = ttest_rel(complementary_leader_stack, complementary_sync_egal_stack, axis=2, nan_policy = 'omit')




    for freq_band_idx in range(num_freq_bands):
        
        # Select the correct axes for the current subplot
        ax = axs[contrast_idx, freq_band_idx]
           
    
        
        t_stat_freq = t_stats[:, freq_band_idx]
        # Create the EvokedArray object
        # The data needs to be in the shape (n_channels, n_times), so we add an extra dimension to power_at_freq_band
        #evoked = mne.EvokedArray(power_at_freq_band[:, np.newaxis], info)
        evoked = mne.EvokedArray(t_stat_freq[:, np.newaxis], info)

        # Plot the topomap for the Evoked object on the specified axes
        # Note: For actual use, ensure 'times' parameter matches your Evoked data's time point(s)
        #mne.viz.plot_topomap(power_at_freq_band, evoked.info, axes=ax, show=False, contours=0,
        #                     sensors=False, res=64, names=evoked.ch_names)
        
        #mne.viz.plot_topomap(t_stat_freq, evoked.info, axes=ax, show=False, contours=2,
         #                    sensors=False, res=64, names=evoked.ch_names)
        
        # Optionally, set titles, etc.
        ax.set_title(f'Contrast {contrast_idx}, Band {freq_band_idx}')
        ax.plot(t_stat_freq)
# Show the complete figure
plt.show()

# Create a figure with a grid of subplots
fig, axs = plt.subplots(num_contrasts, num_freq_bands, figsize=(20, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
# Iterate over each contrast and frequency band to create and plot Evoked objects
for contrast_idx in range(num_contrasts):
    if contrast_idx == 0:
        participant_wise_contrast = sync_egal_stack - individual_stack
        t_stats, p_values = ttest_rel(sync_egal_stack, individual_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 1:
        participant_wise_contrast = leader_stack - follower_stack
        t_stats, p_values = ttest_rel(leader_stack, follower_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 2:
        participant_wise_contrast = complementary_sync_egal_stack - sync_egal_stack
        t_stats, p_values = ttest_rel(complementary_sync_egal_stack, sync_egal_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 3:
        participant_wise_contrast = complementary_leader_stack - complementary_follower_stack
        t_stats, p_values = ttest_rel(complementary_leader_stack, complementary_follower_stack, axis=2, nan_policy = 'omit')


    # Initialize a container for the cleaned contrast data
    cleaned_contrast = np.empty(participant_wise_contrast.shape)
    cleaned_contrast[:] = np.nan  # Fill with NaNs to later ignore these in mean calculation

    # Iterate over each channel and frequency band to exclude outliers
    for ch_idx in range(participant_wise_contrast.shape[0]):
        for freq_band in range(participant_wise_contrast.shape[1]):
            data = participant_wise_contrast[ch_idx, freq_band, :]
            mean = np.mean(data)
            std = np.std(data)
            # Identify outliers
            outliers_mask = np.abs(data - mean) > 3 * std
            # Exclude outliers by setting them to NaN
            data[outliers_mask] = np.nan
            # Store the cleaned data
            cleaned_contrast[ch_idx, freq_band, :] = data

    # Calculate the average across participants, ignoring NaN values
    average_contrast = np.nanmean(cleaned_contrast, axis=2)



    for freq_band_idx in range(num_freq_bands):
        
        # Select the correct axes for the current subplot
        ax = axs[contrast_idx, freq_band_idx]
           
    
        
        power_at_freq_band = average_contrast[:, freq_band_idx]
        t_stat_freq = t_stats[:, freq_band_idx]
        # Create the EvokedArray object
        # The data needs to be in the shape (n_channels, n_times), so we add an extra dimension to power_at_freq_band
        #evoked = mne.EvokedArray(power_at_freq_band[:, np.newaxis], info)
        evoked = mne.EvokedArray(t_stat_freq[:, np.newaxis], info)

        # Plot the topomap for the Evoked object on the specified axes
        # Note: For actual use, ensure 'times' parameter matches your Evoked data's time point(s)
        #mne.viz.plot_topomap(power_at_freq_band, evoked.info, axes=ax, show=False, contours=0,
        #                     sensors=False, res=64, names=evoked.ch_names)
        
        mne.viz.plot_topomap(t_stat_freq, evoked.info, axes=ax, show=False, contours=2,
                            sensors=True, res=64, names=evoked.ch_names)
        
        # Optionally, set titles, etc.
        ax.set_title(f'Contrast {contrast_idx}, Band {freq_band_idx}')


# Create a figure with a grid of subplots
fig, axs = plt.subplots(num_contrasts, num_freq_bands, figsize=(20, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots

# Iterate over each contrast and frequency band to create and plot Evoked objects
for contrast_idx in range(num_contrasts):
    if contrast_idx == 0:
        participant_wise_contrast = sync_egal_stack - individual_stack
        t_stats, p_values = ttest_rel(sync_egal_stack, individual_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 1:
        participant_wise_contrast = leader_stack - follower_stack
        t_stats, p_values = ttest_rel(leader_stack, follower_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 2:
        participant_wise_contrast = complementary_sync_egal_stack - sync_egal_stack
        t_stats, p_values = ttest_rel(complementary_sync_egal_stack, sync_egal_stack, axis=2, nan_policy = 'omit')
    elif contrast_idx == 3:
        participant_wise_contrast = complementary_leader_stack - complementary_follower_stack
        t_stats, p_values = ttest_rel(complementary_leader_stack, complementary_follower_stack, axis=2, nan_policy = 'omit')


    # Initialize a container for the cleaned contrast data
    cleaned_contrast = np.empty(participant_wise_contrast.shape)
    cleaned_contrast[:] = np.nan  # Fill with NaNs to later ignore these in mean calculation

    # Iterate over each channel and frequency band to exclude outliers
    for ch_idx in range(participant_wise_contrast.shape[0]):
        for freq_band in range(participant_wise_contrast.shape[1]):
            data = participant_wise_contrast[ch_idx, freq_band, :]
            mean = np.mean(data)
            std = np.std(data)
            # Identify outliers
            outliers_mask = np.abs(data - mean) > 3 * std
            # Exclude outliers by setting them to NaN
            data[outliers_mask] = np.nan
            # Store the cleaned data
            cleaned_contrast[ch_idx, freq_band, :] = data

    # Calculate the average across participants, ignoring NaN values
    average_contrast = np.nanmean(cleaned_contrast, axis=2)



    for freq_band_idx in range(num_freq_bands):
        
        # Select the correct axes for the current subplot
        ax = axs[contrast_idx, freq_band_idx]
           
    
        
        power_at_freq_band = average_contrast[:, freq_band_idx]
        t_stat_freq = t_stats[:, freq_band_idx]
        # Create the EvokedArray object
        # The data needs to be in the shape (n_channels, n_times), so we add an extra dimension to power_at_freq_band
        #evoked = mne.EvokedArray(power_at_freq_band[:, np.newaxis], info)
        evoked = mne.EvokedArray(t_stat_freq[:, np.newaxis], info)

        # Plot the topomap for the Evoked object on the specified axes
        # Note: For actual use, ensure 'times' parameter matches your Evoked data's time point(s)
        #mne.viz.plot_topomap(power_at_freq_band, evoked.info, axes=ax, show=False, contours=0,
        #                     sensors=False, res=64, names=evoked.ch_names)
        
        #mne.viz.plot_topomap(t_stat_freq, evoked.info, axes=ax, show=False, contours=2,
         #                    sensors=False, res=64, names=evoked.ch_names)
        
        # Optionally, set titles, etc.
        print(np.shape(participant_wise_contrast))
        ax.set_title(f'Contrast {contrast_idx}, Band {freq_band_idx}')
        ax.plot(average_contrast[:, freq_band_idx])
# Show the complete figure
plt.show()
