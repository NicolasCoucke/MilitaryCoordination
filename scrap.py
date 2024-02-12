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
import hypyp
import requests
import os
import PyQt5
import sys
import pickle
#print(sys.path)
sys.path.append('C:/Users/Administrateur/MilitaryCoordination/')

from hypyp import (
    prep,)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import prep
from hypyp import stats
from hypyp import viz
import autoreject

import copy
import my_utils
from my_utils import extract_trials, create_sub_epochs, AR_local_custom, ICA_autocorrect, AR_global_custom


path = r"F:/hyperscanning_mne"
raw_path = r"F:/Hyperscanning_eeg_data"
prep_path = os.path.join(path, "preprocessed data")
log_path = os.path.join(path, "logs")

for pair in range(1,45):
    file_path = os.path.join("F:/hyperscanning_mne", "pair_" + str(pair))

    try: 
        with open(file_path , "rb") as input_file:
            cleaned_epochs_AR, dic_AR = pickle.load(input_file)
    except:
        print('nothing')
        # continue



    #cleaned_epochs_AR, dic_AR, bad_epochs_AR = AR_global_custom(cleaned_epochs_AR, strategy = 'union', threshold = 50.0, verbose = False)


    print('apply extra segment rejection step')
    bad_labels = []
    rejection_ratios = []
    for cleaned_epochs in cleaned_epochs_AR:
        data = cleaned_epochs.get_data()

        # create label matrix
        n_epochs = np.size(data, 0)
        n_channels = np.size(data, 1)
        n_times = np.size(data, 2)
        label_array = np.zeros((n_epochs, n_channels))

        # for the whole data, compute the frequency spectrum and delete channels that have more than 2 SD of power than the other channels on average
        # later try also with fixed threshold

        #psds, freq = mne.time_frequency.psd_multitaper(cleaned_epochs, fmin = 4, fmax = 45, njobs = 1)
        psds = cleaned_epochs.compute_psd(fmin = 4, fmax = 45).get_data()
        psds = np.mean(psds, 0)
        channel_powers = []
        for channel_index in range(64):
            psd_channel = psds[channel_index,:]
            average_power = np.mean(psd_channel)
            channel_powers.append(average_power*10**12) # microvolt squared
            plt.plot(psd_channel*10**12)
        plt.show(block = True)
        
        channel_powers = np.array(channel_powers)
        rejected_channels = np.where((np.abs(channel_powers - np.mean(channel_powers)) > 2*np.std(channel_powers)) or (channel_powers > 250)) #np.where(channel_powers > 200) #np.where(np.abs(channel_powers - np.mean(channel_powers)) > 2*np.std(channel_powers))
        
        print(rejected_channels)
        # reject the channels in the index
        for channel in rejected_channels[0]:
            label_array[:, channel] = np.ones(n_epochs)





        
        threshold = 100 # microvolt

        standard_dev = np.std(data, axis = 2)
        mean_std = np.mean(standard_dev)
        std_std = np.std(standard_dev)
        std_threshold = mean_std + 2*std_std

        
        for epoch in range(n_epochs):
            for channel in range(n_channels):
                ptp = np.max(data[epoch, channel,:]) - np.min(data[epoch, channel,:])
                std = np.std(data[epoch, channel,:])
                if ptp > threshold*1e-06:
                    label_array[epoch, channel] = 1
                if std > std_threshold:
                    label_array[epoch, channel] = 1

            if np.sum(label_array[epoch,:]) > 6:
                label_array[epoch,:] = np.ones(64)

        # reject the whole trial if there is too much out
        # 
                    
        plt.imshow(np.transpose(label_array))
        plt.show(block = True)
        print(channel_powers)

        rejection_ratio = np.sum(label_array) / np.size(label_array)
        print('proportion_rejected =' + str(rejection_ratio))
        bad_labels.append(label_array)
        rejection_ratios.append(rejection_ratio)

        

    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]




# after rereferencing, we can reject channels that are still above 25dB
raws = [raw_1, raw_2]
print('start rejection after reref')
part = 1
for epochs_i, raw_i in zip(epochs, raws):
    data = epochs_i.get_data()
    
    # create label matrix
    n_epochs = np.size(data, 0)
    n_channels = np.size(data, 1)
    n_times = np.size(data, 2)
    label_array = np.zeros((n_epochs, n_channels))
    channel_powers = []
    psds, freqs = raw_i.compute_psd(fmin = 1., fmax = 50., n_fft = 512).get_data(return_freqs = True)
    print(psds.shape)
    for channel_index in range(64):
        psd_channel = 10 * np.log10(psds[channel_index,:]*10**12)
        average_power = np.mean(psd_channel)
        channel_powers.append(average_power) # microvolt squared

    
    # detect channel powers that are too large
    channel_powers = np.array(channel_powers)
    rejected_channels_threshold = np.where(channel_powers > 25) # removed this to avoid problems when the mastoids make the whole recording noisy
    #rejected_channels = np.unique(np.concatenate((rejected_channels_std[0], rejected_channels_threshold[0])))
    rejected_channels = rejected_channels_threshold[0]
    rejected_channels.astype(int)

    # mark those channels as bad in the raw object
    channel_names = list(mapping_1.values())
    rejected_channel_names = [channel_names[i] for i in rejected_channels.tolist()]
    raw_i.info['bads'] = rejected_channel_names

    # reject the channels in the index
    for channel in rejected_channels:
        label_array[:, channel] = np.ones(n_epochs)

    rejected_channels_pair[part-1] = np.concatenate((rejected_channels_pair[part-1], rejected_channels))
    
    fig = plt.figure()
    plt.imshow(np.transpose(label_array), aspect = 'auto')
    ax = plt.gca()
    ax.grid(which = 'minor', color = 'w', linestyle = '-', linewidth = 1)
    plt.savefig(os.path.join(log_folder_path, 'post_rref_label_array_' + str(part)))
    part+=1

# and now interpolate the bad channels
raw_1.interpolate_bads()
raw_2.interpolate_bads()




"""
path = r"F:/hyperscanning_mne"
for pair in range(1,45):

    if (pair) > 20:
        group = 'civilian'
    else:
        group = 'military'

    ######### load in files ##############
    
    try:

        file_path = os.path.join("F:/hyperscanning_mne/", "pair_" + str(pair))     
        with open(file_path , "rb") as input_file:
            cleaned_epochs_AR, dic_AR = pickle.load(input_file)
                
    except:
        print('pair not available')
        continue

    path = r"F:/hyperscanning_mne"
    log_path = os.path.join(path, "logs")
    log_folder_path = os.path.join(log_path, 'pair_' + str(pair) + '_')
    
   
    ###### get positions of conditions for extra rejection steps##############
    event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}
    epoch_position_dict = {}

    print('calculate complex signal for all conditions')
    for condition in event_id.keys():
        print(condition)
        if event_id[condition] in cleaned_epochs_AR[0].events[:,2]:
            print(cleaned_epochs_AR[0][condition])
            fig = mne.Epochs.compute_psd(cleaned_epochs_AR[0][condition]).plot_topomap()
            plt.savefig(log_folder_path + 'person_1_' + 'post_ar_freq_'+ str(event_id[condition]))
            fig = mne.Epochs.compute_psd(cleaned_epochs_AR[1][condition]).plot_topomap()
            plt.savefig(log_folder_path + 'person_2_' + 'post_ar_freq_'+ str(event_id[condition]))
    plt.close('all')

"""