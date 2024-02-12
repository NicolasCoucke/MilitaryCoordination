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
from my_utils import extract_trials, create_sub_epochs, AR_local_custom, ICA_autocorrect


path = r"F:/hyperscanning_mne"
raw_path = r"F:/Hyperscanning_eeg_data"
prep_path = os.path.join(path, "preprocessed data")
log_path = os.path.join(path, "logs")



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

            if not (pair in range(37,45)):
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
            events, event_successes = create_sub_epochs(trials, sfreq)
            #mne.viz.plot_events(events, sfreq=raw.info['sfreq']);
            print("fs" + str(raw.info['sfreq']))
            event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}

            log_path = os.path.join(path, "logs")
            log_folder_path = os.path.join(log_path, name)


            fig = mne.viz.plot_raw_psd(raw_1, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'raw_1_spectrum'))

            raw_1.notch_filter(freqs = [50, 100, 150, 200, 250], filter_length='auto', method = 'fir', trans_bandwidth = 4, verbose = False)
            fig = mne.viz.plot_raw_psd(raw_1, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'notch_1_spectrum'))

            raw_1.filter(1, 45, verbose = False)
            fig = mne.viz.plot_raw_psd(raw_1, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'bandpass_1_spectrum'))

            raw_1.set_eeg_reference(ref_channels='average', verbose = False)
            fig = mne.viz.plot_raw_psd(raw_1, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'reref_1_spectrum'))

            # pass the events in the function os they are also resampled
            raw1, events = raw_1.resample(sfreq = 512, verbose = False, events = events)
            fig = mne.viz.plot_raw_psd(raw_1, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'resamp_1_spectrum'))


            fig = mne.viz.plot_raw_psd(raw_2, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'raw_2_spectrum'))

            raw_2.notch_filter(freqs = [50, 100, 150, 200, 250], filter_length='auto', method = 'fir', trans_bandwidth = 4, verbose = False)
            fig = mne.viz.plot_raw_psd(raw_1, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'notch_2_spectrum'))

            raw_2.filter(1, 45, verbose = False)
            fig = mne.viz.plot_raw_psd(raw_2, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'bandpass_2_spectrum'))

            raw_2.set_eeg_reference(ref_channels='average', verbose = False)
            fig = mne.viz.plot_raw_psd(raw_2, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'reref_2_spectrum'))

            # for the second participant, pass the succes event variable for convenience
            raw2, event_successes = raw_2.resample(sfreq = 512, verbose = False, events = event_successes)
            fig = mne.viz.plot_raw_psd(raw_2, area_mode='range', average=False)
            plt.savefig(os.path.join(log_folder_path, 'resamp_2_spectrum'))

            plt.close('all')

            epochs_1 = mne.Epochs(raw_1, events, event_id, event_repeated = 'drop', on_missing = 'ignore', tmin=0, tmax=1, preload=True, baseline=(0, 0))
            epochs_2 = mne.Epochs(raw_2, events, event_id, event_repeated = 'drop', on_missing = 'ignore', tmin=0, tmax=1, preload=True, baseline=(0, 0))

            fig = mne.Epochs.plot(epochs_1, block = False)
            plt.savefig(os.path.join(log_folder_path, 'pre_ica_1_epochs'))
            fig = mne.Epochs.plot(epochs_2, block = False)
            plt.savefig(os.path.join(log_folder_path, 'pre_ica_2_epochs'))

            icas = prep.ICA_fit([epochs_1, epochs_2],
                    n_components=32,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state= 42)

            cleaned_epochs_ICA = [epochs_1, epochs_2]
            #cleaned_epochs_ICA = prep.ICA_choice_comp(icas, cleaned_epochs_ICA)
            cleaned_epochs_ICA, n_ICA_excluded = ICA_autocorrect(icas, [epochs_1, epochs_2], verbose=True)

            fig = mne.Epochs.plot(cleaned_epochs_ICA[0])
            plt.savefig(os.path.join(log_folder_path, 'post_ica_1_epochs'))

            fig = mne.Epochs.plot(cleaned_epochs_ICA[1])
            plt.savefig(os.path.join(log_folder_path, 'post_ica_2_epochs'))

            plt.close('all')

            #n_interpolates = np.array([1, 2, 3, 4, 5, 6])
            #consensus_percs = np.linspace(0, 0.5, 11)
            n_interpolates = np.array([1, 2, 3, 4])
            consensus_percs = np.linspace(0, 1.0, 11)

            before_cleaning = copy.deepcopy(cleaned_epochs_ICA)
            cleaned_epochs_AR, dic_AR, bad_epochs_AR  = AR_local_custom(cleaned_epochs_ICA, n_interpolates, consensus_percs,
                                                    strategy="union",
                                                    threshold=50.0,
                                                    verbose=False
            )


            pair_info = dict()
            pair_info['n_ICA_excluded'] = n_ICA_excluded
            pair_info['dic_AR'] = dic_AR
            pair_info['bad_epochs_AR'] = bad_epochs_AR
            
            fig = mne.Epochs.plot(cleaned_epochs_AR[0], block = False)
            fig.savefig(os.path.join(log_folder_path, '_after_AR_1'))
            plt.close('all')

            fig = mne.Epochs.plot(cleaned_epochs_AR[1], block = False)
            fig.savefig(os.path.join(log_folder_path,  '_after_AR_2'))
            plt.close('all')



            #fig = bad_epochs_AR[0].plot(block = False)
           # bad_epochs_AR[0].plot()
           # fig.savefig(os.path.join(log_folder_path, 'AR_rejection_log_1'))
            plt.close('all')
            #fig = bad_epochs_AR[1].plot(block = False)
           # bad_epochs_AR[0].plot()             
           # fig.savefig(os.path.join(log_folder_path, 'AR_rejection_log_2'))

            plt.close('all')
            #pickle save the cleaned epochs and make sure all the data that we want is in there
            storepath = os.path.join("F:/hyperscanning_mne", "pair_" + str(pair))

            with open(storepath, "wb") as output_file: 
                pickle.dump([cleaned_epochs_AR, pair_info], output_file, protocol=pickle.HIGHEST_PROTOCOL)
            pair +=1

            plt.close('all')