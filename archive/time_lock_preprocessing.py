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
from my_utils import extract_trials, create_sub_epochs, AR_local_custom, ICA_autocorrect, create_time_locked_epochs


path = r"F:/hyperscanning_mne"
raw_path = r"F:/Hyperscanning_eeg_data"
prep_path = os.path.join(path, "preprocessed data")
log_path = os.path.join(path,"time_locked", "logs")

"""
Here we extract trials that are then time-locked to the sync or desync points
the only difference with script 'preprocessing2' is the 'create_time_locked_epochs' function and the lenght of the epochs (-1000 to 500)

We extract 1 epoch at the start of each trial and then one for each checkpoint



"""

pre_stim = 1.5
post_stim = 1

pair = 1
for root, dirs, files in os.walk(raw_path):
    for name in files:
        if name.endswith((".bdf")):
            print("processing file " + name)
            file_path = os.path.join(raw_path, name)
            log_folder_path = os.path.join(log_path, name)
            if not os.path.isdir(log_folder_path):
                os.makedirs(log_folder_path)

            split_name = name.split("P")
            pair = int ((int(split_name[1]) + 1) / 2)

            if not (pair in range(1,45)):
                continue
            #if (pair == 33) or (pair == 28) or (pair == 15) or (pair == 12):
            #   continue

            print('beyond')
            raw = mne.io.read_raw_bdf(file_path, preload = True)
            channels_1 = raw.info.ch_names[:64]
            raw_1 = raw.pick_channels(channels_1)
            del raw
            raw = mne.io.read_raw_bdf(file_path, preload = True)
            channels_2 = raw.info.ch_names[76:140]
            raw_2 = raw.pick_channels(channels_2)
            del raw

                        # get the montage that we will use
            biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
            #biosemi64_montage.plot()  # 2D

            # change the channel names in our epochs so that they are the same as the montage
            channels = biosemi64_montage.ch_names
            mapping_1 = {}
            mapping_2 = {}
            for i in range(len(channels)):
                mapping_1[raw_1.info['chs'][i]['ch_name']] = str(channels[i])
                mapping_2[raw_2.info['chs'][i]['ch_name']] = str(channels[i])
            mne.rename_channels(raw_1.info, mapping_1)
            mne.rename_channels(raw_2.info, mapping_2)

            raw_1.pick_channels(channels)
            raw_2.pick_channels(channels)


            # set the montage to the epochs
            raw_1.set_montage(biosemi64_montage)
            raw_2.set_montage(biosemi64_montage)

            raw = mne.io.read_raw_bdf(file_path)
            raw_events = mne.find_events(raw, shortest_event = 0)
            trials = extract_trials(raw_events)
            sfreq = raw.info['sfreq']
            events, event_info = create_time_locked_epochs(trials, sfreq, raw_events)  

            print(events)
            print(event_info)
            
            #mne.viz.plot_events(events, sfreq=raw.info['sfreq']);
            print("fs" + str(raw.info['sfreq']))
            event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}

            log_path = os.path.join(path, "logs")
            log_folder_path = os.path.join(log_path, name)


            raw_1.notch_filter(freqs = [50, 100, 150, 200, 250], filter_length='auto', method = 'fir', trans_bandwidth = 4, verbose = False)


            raw_1.filter(1, 45, verbose = False)
            fig = mne.viz.plot_raw_psd(raw_1, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'after_filtering_1_spectrum'))



            fig = mne.viz.plot_raw_psd(raw_2, area_mode='range', average=False)


            raw_2.notch_filter(freqs = [50, 100, 150, 200, 250], filter_length='auto', method = 'fir', trans_bandwidth = 4, verbose = False)


            raw_2.filter(1, 45, verbose = False)
            fig = mne.viz.plot_raw_psd(raw_2, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'after_filtering_2_spectrum'))

            plt.close('all')

            # now create the epochs a first time to get the dimensions and also to make a butterfly plot
            epochs_1 = mne.Epochs(raw_1, events, event_id, event_repeated = 'drop', on_missing = 'ignore', tmin=-pre_stim, tmax=post_stim, preload=True, baseline=(0, 0))
            epochs_2 = mne.Epochs(raw_2, events, event_id, event_repeated = 'drop', on_missing = 'ignore', tmin=-pre_stim, tmax=post_stim, preload=True, baseline=(0, 0))

            # make now a plot of the epochs before channel rejection, rereferencing and resmapling
            evoked_1 = epochs_1.average()
            fig = mne.viz.plot_evoked(evoked_1, show = False)
            plt.savefig(os.path.join(log_folder_path, 'post_filtering_1_epochs'))
            evoked_2 = epochs_2.average()
            fig = mne.viz.plot_evoked(evoked_2, show = False)
            plt.savefig(os.path.join(log_folder_path, 'post_filtering_2_epochs'))

            epochs = [epochs_1, epochs_2]
            raws = [raw_1, raw_2]
            bad_labels = []
            rejection_ratios = []
            rejected_channels_pair = []

            print('start rejection before reref')
            part = 1
            for epochs_i, raw_i in zip(epochs, raws):
                data = epochs_i.get_data()

                # create label matrix
                n_epochs = np.size(data, 0)
                n_channels = np.size(data, 1)
                n_times = np.size(data, 2)
                label_array = np.zeros((n_epochs, n_channels))

                # for the whole data, compute the frequency spectrum and delete channels that have more than 2 SD of power than the other channels on average
                # later try also with fixed threshold
                channel_powers = []
                psds, freqs = raw_i.compute_psd(fmin = 1., fmax = 50., n_fft = 2048).get_data(return_freqs = True)
                print(psds.shape)
                for channel_index in range(64):
                    psd_channel = 10 * np.log10(psds[channel_index,:]*10**12)
                    average_power = np.mean(psd_channel)
                    channel_powers.append(average_power) # microvolt squared

                
                # detect channel powers that are too large
                channel_powers = np.array(channel_powers)
                rejected_channels_std = np.where((np.abs(channel_powers - np.mean(channel_powers)) > 2*np.std(channel_powers))) #np.where(channel_powers > 200) #np.where(np.abs(channel_powers - np.mean(channel_powers)) > 2*np.std(channel_powers))
                #rejected_channels_threshold = np.where(channel_powers > 25) # removed this to avoid problems when the mastoids make the whole recording noisy
                #rejected_channels = np.unique(np.concatenate((rejected_channels_std[0], rejected_channels_threshold[0])))
                rejected_channels = rejected_channels_std[0]
                rejected_channels.astype(int)

                # mark those channels as bad in the raw object
                channel_names = list(mapping_1.values())
                rejected_channel_names = [channel_names[i] for i in rejected_channels.tolist()]
                raw_i.info['bads'] = rejected_channel_names

                # reject the channels in the index
                for channel in rejected_channels:
                    label_array[:, channel] = np.ones(n_epochs)

                rejected_channels_pair.append(rejected_channels)
                
                fig = plt.figure()
                plt.imshow(np.transpose(label_array), aspect = 'auto')
                ax = plt.gca()
                ax.grid(which = 'minor', color = 'w', linestyle = '-', linewidth = 1)
                plt.savefig(os.path.join(log_folder_path, 'label_array_' + str(part)))
                part+=1

            # and now interpolate the bad channels
            raw_1.interpolate_bads()
            raw_2.interpolate_bads()
            
            # now, rereference and resample
            raw_1.set_eeg_reference(ref_channels='average', verbose = False)
            # pass the events in the function os they are also resampled
            raw1, events = raw_1.resample(sfreq = 512, verbose = False, events = events)

            # rereference and resample person 2
            raw_2.set_eeg_reference(ref_channels='average', verbose = False)
            # for the second participant, pass the succes event variable for convenience
            raw2, events = raw_2.resample(sfreq = 512, verbose = False, events = events)

            fig = mne.viz.plot_raw_psd(raw_1, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'after_interpolation_1_spectrum'))

            fig = mne.viz.plot_raw_psd(raw_2, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'after_interpolation_2_spectrum'))

            
            # now divide them back into epochs
            epochs_1 = mne.Epochs(raw_1, events, event_id, event_repeated = 'drop', on_missing = 'ignore', tmin=-pre_stim, tmax=post_stim, preload=True, baseline=(0, 0))
            epochs_2 = mne.Epochs(raw_2, events, event_id, event_repeated = 'drop', on_missing = 'ignore', tmin=-pre_stim, tmax=post_stim, preload=True, baseline=(0, 0))

            # make new butterfly plot to see effect of rejected channels
            evoked_1 = epochs_1.average()
            fig = mne.viz.plot_evoked(evoked_1, show = False)
            plt.savefig(os.path.join(log_folder_path, 'post_interpolation_1_epochs'))
            evoked_2 = epochs_2.average()
            fig = mne.viz.plot_evoked(evoked_2, show = False)
            plt.savefig(os.path.join(log_folder_path, 'post_interpolation_2_epochs'))

            # now the bad channels have been interpolated and we can do the ica on the interpolated data
            icas = prep.ICA_fit([epochs_1, epochs_2],
                    n_components=32,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state= 42)

            cleaned_epochs_ICA, n_ICA_excluded = ICA_autocorrect(icas, [epochs_1, epochs_2], verbose=True)

            # plot the butterflies after the bad icas have been rejected
            evoked_1 = cleaned_epochs_ICA[0].average()
            fig = mne.viz.plot_evoked(evoked_1, show = False)
            plt.savefig(os.path.join(log_folder_path, 'post_autocorrect_1_epochs'))
            evoked_2 = cleaned_epochs_ICA[1].average()
            fig = mne.viz.plot_evoked(evoked_2, show = False)
            plt.savefig(os.path.join(log_folder_path, 'post_autocorrect_2_epochs'))




            pair_info = dict()
            pair_info['n_ICA_excluded'] = n_ICA_excluded
            pair_info['bad_channels'] = rejected_channels_pair
            pair_info['events'] = events
            pair_info['event_info'] = event_info
            
            plt.close('all')
            #pickle save the cleaned epochs and make sure all the data that we want is in there
            storepath = os.path.join("F:/hyperscanning_mne", "time_locked","pair_" + str(pair))
            with open(storepath, "wb") as output_file: 
                pickle.dump([cleaned_epochs_ICA, pair_info], output_file, protocol=pickle.HIGHEST_PROTOCOL)
            pair +=1

            plt.close('all')
