from copy import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import mne
#import hypyp
import requests
import os
import PyQt5
import sys
import pickle

import autoreject
from mne_icalabel import label_components

import copy
import my_utils
from my_utils import extract_trials, create_sub_epochs, AR_local_custom, ICA_autocorrect, AR_global_custom, link_eeg_to_behavioral_trials, get_channels_to_reject


path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1"
raw_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\raw data"
prep_path = os.path.join(path, "preprocessed data")
log_path = os.path.join(path, "logs")



# loop through all data files
pair = 1
for root, dirs, files in os.walk(raw_path):
    for name in files:
        if name.endswith((".bdf")):
            print("processing file " + name)

            # define paths
            file_path = os.path.join(raw_path, name)
            log_folder_path = os.path.join(log_path, name)
            if not os.path.isdir(log_folder_path):
                os.makedirs(log_folder_path)

            split_name = name.split("P")
            pair = int ((int(split_name[1]) + 1) / 2)

            # read in data
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


            #notch and band pass filtering
            raw_1.notch_filter(freqs = [50, 100, 150, 200, 250], filter_length='auto', method = 'fir', trans_bandwidth = 4, verbose = False)
            raw_1.filter(1, 45, verbose = False)

            raw_2.notch_filter(freqs = [50, 100, 150, 200, 250], filter_length='auto', method = 'fir', trans_bandwidth = 4, verbose = False)
            raw_2.filter(1, 45, verbose = False)

            # extract trials 
            raw = mne.io.read_raw_bdf(file_path)
            events = mne.find_events(raw, shortest_event = 0)
            trials = extract_trials(events)
            
            # link the eeg and behavioral data
            with open(r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\behavioral data\Behavioral_Dataframe.pickle", "rb") as input_file:
                data_dictionary = pickle.load(input_file)
            updated_trials = link_eeg_to_behavioral_trials(trials, data_dictionary, pair, sfreq)

            # only keep trials for which we found behavioral data
            updated_trials = updated_trials[np.where(~np.isnan(updated_trials[:,4]))[0],:]


            # make epochs from trials
            sfreq = raw_1.info['sfreq']
            events, event_create_sub_epochs= create_sub_epochs(trials, sfreq, 2, 0.5)
            print("fs" + str(raw.info['sfreq']))
            event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}

            # splice out all the data that we don't need
            #new_trial = [trial_begin, trial_end, trial_counter, condition, trialnumber]
            segments = updated_trials[:,:2]

            raws = [raw_1, raw_2]
            spliced_raws = []
            for raw_i in raws:
                cropped_segments = []
                # Loop through each segment and crop the raw data
                for segment in segments:
                    start_index, end_index = segment
                    
                    # Get the corresponding start and end times using the sample indices
                    start_time = raw_i.times[int(start_index-2)]
                    end_time = raw_i.times[int(end_index+2)]

                    # Crop the raw data using the obtained times
                    cropped_segment = raw_i.copy().crop(tmin=start_time, tmax=end_time)
                    cropped_segments.append(cropped_segment)

                # Concatenate the cropped segments back together
                spliced_raw = mne.concatenate_raws(cropped_segments)
                spliced_raws.append(spliced_raw)

            # reject very bad channels and then perform ICA:
            mappings = [mapping_1, mapping_2]

            raws_ICA_applied = []
            # interpolate bad channels but 
            map = 0
            for spliced_raw in spliced_raws:
                
                # detect very bad channels
                rejected_channels = get_channels_to_reject(spliced_raw, events)
                channel_names = list(mappings[map].values())
                rejected_channel_names = [channel_names[i] for i in rejected_channels.tolist()]
                channels_to_reject = rejected_channel_names

                # interpolate them
                spliced_raw.info['bads'] = channels_to_reject
                spliced_raw.interpolate_bads()
                
                # set eeg reference after interpolation
                spliced_raw.set_eeg_reference(ref_channels='average', verbose = False)
     
                
                # Create a copy of the raw data and remove the interpolated channels from the copy
                raw_copy = spliced_raw.copy().drop_channels(channels_to_reject)

                # Perform ICA on the copy without the interpolated channels
                ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
                ica.fit(raw_copy)

                # reconstruct the data without the non-brain components
                ica_with_labels_fitted = label_components(raw_copy, ica, method="iclabel")
                ica_with_labels_component_detected = ica_with_labels_fitted["labels"]               
                excluded_idx_components = [idx for idx, label in enumerate(ica_with_labels_component_detected) if label not in ["brain"]]
                raw_ica_applied = ica.apply(raw_copy.copy(), exclude=excluded_idx_components)


                # Interpolate the removed channels on the raw data with ICA applied
                raw_ica_applied.interpolate_bads(reset_bads=False)
                raws_ICA_applied.append(raw_ica_applied)
                map+=1

            # now epoch the data and perform autoreject        
            epochs = []

            for raw in raws_ICA_applied:
                epoch = mne.Epochs(raw, events, event_id, event_repeated = 'drop', on_missing = 'ignore', tmin=0, tmax=2, preload=True, baseline=(0, 0))


            n_interpolates = np.array([1, 2, 3, 4])
            consensus_percs = None

            cleaned_epochs_AR, dic_AR, bad_epochs_AR  = AR_local_custom(epochs, n_interpolates, consensus_percs,
                                                    strategy="union",
                                                    threshold=50.0,
                                                    verbose=False
            )

            # save the data
            storepath = os.path.join(prep_path,"pair_" + str(pair))
            with open(storepath, "wb") as output_file:
                pickle.dump(cleaned_epochs_AR, output_file, protocol=pickle.HIGHEST_PROTOCOL)
