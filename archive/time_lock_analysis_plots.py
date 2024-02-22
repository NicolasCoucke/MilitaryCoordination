import pandas as pd

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
import copy
import pickle
import statsmodels.api as sm
import autoreject
from statsmodels.formula.api import ols
#print(sys.path)
sys.path.append('C:/Users/Administrateur/MilitaryCoordination/')

from hypyp import (
    prep,)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import prep
from hypyp import stats
from hypyp import viz




biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
# change the channel names in our epochs so that they are the same as the montage
channels = biosemi64_montage.ch_names
freq_bands = {'Theta': [4, 7],
            'Alpha': [8, 12],
            'Beta': [13, 30],
            'Gamma': [30, 45]}

freq_list = ['Theta', 'Alpha', 'Beta', 'Gamma']

ROI_channel_names = dict()
ROI_channel_names["left_frontal"] = ["Fp1", "AF7", "AF3", "F7", "F5", "F3"]
ROI_channel_names["right_frontal"] = ["Fp2", "AF8", "AF4", "F8", "F6", "F4"]

ROI_channel_names["left_central"] = ["FC1", "FC3", "C3", "C1", "CP3","CP1"]
ROI_channel_names["right_central"] = ["FC2", "FC4", "C4", "C2", "CP2","CP4"]

ROI_channel_names["left_temporal"] = ["T7", "TP7", "CP5", "P9", "P7", "P5"]
ROI_channel_names["right_temporal"] = ["T8", "TP8", "CP6", "P10", "P8", "P6"]

ROI_channel_numbers = copy.deepcopy(ROI_channel_names)

for key in ROI_channel_names.keys():
    for i in range(len(ROI_channel_names[key])):
        ch_name = ROI_channel_names[key][i]
        index = channels.index(ch_name)
        ROI_channel_numbers[key][i] = index

ROI_keys = list(ROI_channel_numbers.keys())
print(ROI_keys)

path = r"F:/hyperscanning_mne"
raw_path = r"F:/Hyperscanning_eeg_data"
prep_path = os.path.join(path, "preprocessed data")
log_path = os.path.join(path,"time_locked", "logs")


connectivity_df = pd.DataFrame(columns = ["pair", "group", "condition", "frequency", "ROI_1", "ROI_2", "ROI_combination", "wPLI"])
connectivity_df_successful = pd.DataFrame(columns = ["pair", "group", "condition", "successful", "frequency", "ROI_1", "ROI_2", "ROI_combination", "wPLI"])

event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}
n_ch = 64
i_df = 0
i_df_successful = 0
for pair in range(1,44):

    if (pair) > 20:
        group = 'civilian'
    else:
        group = 'military'

    ######### load in files ##############

        # get the folder in which to store the figures
    for root, dirs, files in os.walk(raw_path):
        for name in files:
            if name.endswith((".bdf")):
                file_path = os.path.join(raw_path, name)
                log_folder_path = os.path.join(log_path, name)
                if not os.path.isdir(log_folder_path):
                    os.makedirs(log_folder_path)

                split_name = name.split("P")
                name_pair = int ((int(split_name[1]) + 1) / 2)
                if pair == name_pair:
                    log_folder_path = os.path.join(log_path, name)
                    break
                else:
                    continue

    
    try:
        file_path = os.path.join("F:/hyperscanning_mne/", "time_locked", "plv_time_locked_connectivity_values_pair_" + str(pair))
        #file_path = os.path.join("F:/hyperscanning_mne", "connectivity_values_pair_" + str(pair))
        print(file_path)
        with open(file_path , "rb") as input_file:
            connectivity_values = pickle.load(input_file) 

        file_path = os.path.join("F:/hyperscanning_mne/", "time_locked","pair_" + str(pair))     
        with open(file_path , "rb") as input_file:
            cleaned_epochs_AR, pair_info = pickle.load(input_file)

        file_path = os.path.join("F:/hyperscanning_mne/","time_locked", "bad_labels_pair_" + str(pair))
        with open(file_path , "rb") as input_file:
            bad_labels = pickle.load(input_file) 



        

                
    except:
        print('pair not available')
        continue
    


    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]


    ###### get positions of conditions for extra rejection steps##############
    event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}
    epoch_position_dict = {}
    epoch_information_dict = {}

    # in this array we find further info of the data 
    event_info = pair_info['event_info']

    for condition in event_id.keys():

        # find the epoch positions of this condition
        epoch_positions = np.where(preproc_S1.events[:,2] == event_id[condition])

        # if this condition exists, store the positions of the data
        if event_id[condition] in preproc_S1.events[:,2]:
            epoch_position_dict[condition] = epoch_positions

            # store also further info 
            epoch_information_dict[condition] = event_info[epoch_positions, :]
            



    ########### apply the extra rejection step to the data ###################
    for condition in connectivity_values.keys():

        try: 
            results = connectivity_values[condition]
        except:
            print('con values not found')
            continue
            
        #print(pair)
        #print(condition)
        #print(epoch_position_dict)
        #print(np.shape(bad_labels[0]))

        # get part of the rejection_log that corresponds to condition
        condition_log_1 = bad_labels[0][epoch_position_dict[condition]]
        condition_log_2 = bad_labels[1][epoch_position_dict[condition]]
        
        
        # set the correct positions of the connectivity matrix to nans
        for ch_1 in range(64):
            # intra-brain
            bad_epochs = np.where((condition_log_1[:,ch_1] == 1))
            results[:,bad_epochs, :, ch_1, ch_1] = np.nan

            for ch_2 in range(64):
                # intra-brain for channel 2
                bad_epochs = np.where((condition_log_2[:,ch_2] == 1))
                results[:,bad_epochs, :, n_ch+ch_2, n_ch+ch_2] = np.nan

                # find bad segments for inter-brain
                bad_epochs = np.clip(condition_log_1[:,ch_1] + condition_log_2[:,ch_2], 0, 1)
                bad_epochs = bad_epochs.astype(int)
                results[:,bad_epochs, :, ch_1, n_ch+ch_2] = np.nan
                results[:,bad_epochs, :, n_ch+ch_2, ch_1] = np.nan

        connectivity_values[condition] = results



    ############### average the connectivity values into ROIs #################################
    ci = 0
    for condition in connectivity_values.keys():
        ci +=1
        condition_data = connectivity_values[condition]
        condition_epoch_info = np.squeeze(epoch_information_dict[condition])
        print(condition_data.shape)
        print(condition_epoch_info.shape)
        # first get the inter_brain part
        n_ch = 64
        condition_data = condition_data[:, :, :, 0:n_ch, n_ch:2*n_ch]

        # loop over all frequencies and inter-brain pairs
        for frequency in range(4):
            freq_string = freq_list[frequency]

            # loop over all ROI's
            for ROI_1 in range(6):
                ROI_1_string = ROI_keys[ROI_1]
                channel_numbers_1 = ROI_channel_numbers[ROI_1_string]
                for ROI_2 in range(6):
                    ROI_2_string = ROI_keys[ROI_2]
                    ROI_combination = ROI_1_string + "-" + ROI_2_string

                    channel_numbers_2 = ROI_channel_numbers[ROI_2_string]


                    

                    # first average within the epochs and then between the epochs
                    n_subs = np.size(condition_data, 2)
                    epoch_list = np.empty((0,n_subs))
                    epoch_list_correct = np.empty((0,n_subs)) # for only correct trials
                    epoch_list_wrong = np.empty((0,n_subs)) # for only incorrect trials

                    for epoch in range(np.shape(condition_data)[1]):
                        epoch_value = 0
                        counter = 0
                        for ch_1 in channel_numbers_1:
                            for ch_2 in channel_numbers_2:
                                if not np.isnan(condition_data[frequency, epoch, :, ch_1, ch_2]).any():
                                    epoch_value+=condition_data[frequency, epoch, :,  ch_1, ch_2]
                                    counter+=1
                        if counter < 9: # if there are less than 9 connections (3x3) in the ROI combination, the epoch should not be taken into account
                            continue
                        
                        # (trial number, successful, baseline)
                        if condition_epoch_info[epoch, 2] == 0:

                            try:
                                epoch_list = np.vstack((epoch_list, epoch_value/counter))
                            except:
                                # if all segments are bad then skip the epoch
                                continue
                                print('no stacking')

                            if condition_epoch_info[epoch, 1] == 1:
                                try:
                                    epoch_list_correct = np.vstack((epoch_list_correct, epoch_value/counter))
                                except:
                                    # if all segments are bad then skip the epoch
                                    continue
                                    print('no stacking')
                            else:
                                try:
                                    epoch_list_wrong = np.vstack((epoch_list_wrong, epoch_value/counter))
                                except:
                                    # if all segments are bad then skip the epoch
                                    continue
                                    print('no stacking')

                    ROI_connectivity = np.nanmean(epoch_list, axis = 0)
                    print(ROI_connectivity)
                  #  plt.plot(ROI_connectivity)
                   # plt.show(block = True)
                    #plt.savefig(os.path.join(log_folder_path, freq_string + str(ci) + ROI_combination))
                    #print(os.path.join(log_folder_path, freq_string + condition + ROI_combination))
                   
                    ROI_connectivity_correct = np.nanmean(epoch_list, axis = 0)
                    ROI_connectivity_wrong = np.nanmean(epoch_list, axis = 0)

                    datapoint = pd.Series(data=[pair, group, condition, 'success', freq_string, ROI_1_string, ROI_2_string, ROI_combination, ROI_connectivity],  index= connectivity_df_successful.columns, name = i_df_successful)             
                    connectivity_df_successful = pd.concat([connectivity_df_successful , datapoint.to_frame().T], ignore_index = True) #grid_results.append(configuration_result)
                    i_df_successful+=1
                    datapoint = pd.Series(data=[pair, group, condition, 'failure', freq_string, ROI_1_string, ROI_2_string, ROI_combination, ROI_connectivity],  index= connectivity_df_successful.columns, name = i_df_successful)            
                    connectivity_df_successful = pd.concat([connectivity_df_successful , datapoint.to_frame().T], ignore_index = True) #grid_results.append(configuration_result)
                    i_df_successful+=1

                    datapoint = pd.Series(data=[pair, group, condition, freq_string, ROI_1_string, ROI_2_string, ROI_combination, ROI_connectivity],  index= connectivity_df.columns, name = i_df)
                                
                    connectivity_df = pd.concat([connectivity_df, datapoint.to_frame().T], ignore_index = True) #grid_results.append(configuration_result)
                    i_df+=1
    connectivity_df.to_csv("time_locked_full_averaged_connectivity_values.csv")
    connectivity_df_successful.to_csv("successful_time_locked_full_averaged_connectivity_values.csv")
print(connectivity_df)