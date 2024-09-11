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
import difflib

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
import pandas as pd
data = []  # Starting with an empty list, but dictionaries can be appended as new rows
df = pd.DataFrame(data)
path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1"

connectivity_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\time locked connectivity"
# loop through all data files
individual_stack = np.zeros((64,44,512))
sync_egal_stack = np.zeros((64,44,512))
leader_stack = np.zeros((64,44,512))
follower_stack = np.zeros((64,44,512))
complementary_sync_egal_stack = np.zeros((64,44,512))
complementary_leader_stack = np.zeros((64,44,512))
complementary_follower_stack = np.zeros((64,44,512))

stack_dict = dict()
military_stack_dict = dict()
condition_names = ['Synchronous/Egalitarian', 'Synchronous/LeaderFollp', 'Synchronous/Follower', 'Complementary/Egalitarian', 'Complementary/Leader', 'Complementary/Follower', 'Start','Individual']
for condition in condition_names:
    condition_stack = []#np.zeros((64,44,512))    #np.zeros((64,44,512))   
    stack_dict[condition] = []
    military_stack_dict[condition] = []

def remove_zero_size_dimensions(matrix):
    """
    Removes dimensions of size zero from a numpy array.

    :param matrix: Input numpy array.
    :return: Numpy array with all zero-size dimensions removed.
    """
    non_zero_dims = [dim for dim in matrix.shape if dim != 0]
    if len(non_zero_dims) == 0:
        return np.array([])  # Return empty array if all dimensions are zero
    return matrix.reshape(non_zero_dims)

     

fronto_central_channels = ['FCz','Cz', 'FC1', 'FC2']

motor_channels = ["C3", "C1", "Cz", "C2", "C4"]

frontal_channels = ["AF3", "AFz", "AF4", "F3", "F1", "Fz", "F2", "F4"]
def extract_freq_channel_combination(stack, freq_values, channels):
    selected_stack = np.nanmean(stack[channels, freq_values[0]:freq_values[1], :,:], axis = (0,1))
    return selected_stack


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


def filter_dict_by_substrings(input_dict, substrings):
    """
    Filter a dictionary to only include entries where the key contains all of the specified substrings.

    :param input_dict: The dictionary to filter.
    :param substrings: A list of substrings to check against dictionary keys.
    :return: A new dictionary with entries that have keys containing all of the substrings.
    """
    # Create a new dictionary using dictionary comprehension
    filtered_dict = {key: value for key, value in input_dict.items()
                     if all(substring in key for substring in substrings)}

    return filtered_dict

pair = 1
for root, dirs, files in os.walk(connectivity_path):
    for name in files:
        if 'trail_based_individual_tfr_' in name:
            print("processing file " + name)

            file_path = os.path.join(connectivity_path, name)
            # define paths
            print(file_path)
            
            split_name = name.split("pair")
            pair = int(split_name[1])

            group = 'none'
            if pair > 20:
                group = 'civilian'
            else:
                group = 'military'
            

            # load in both the individual and the joint data

            with open(file_path,"rb") as input_file:
                participant_1_power_values, participant_2_power_values = pickle.load(input_file)


            
            # also load in the connectivity file
           
            file_path = connectivity_path +  '/trail_based_connectivity_tfr_pair_' + str(pair)

            with open(file_path,"rb") as input_file:
                participant_1_connectivity_values, participant_2_connectivity_values = pickle.load(input_file)
            
            for key, matrix in participant_1_connectivity_values.items():
                print(f'{key}: {matrix.shape}')
            


            # loop through the trials and get the relevant data

            # pair we know

            # condition is first part of the string

            # second part of the string is chekcpoint or trialnumber etc

            power_value_list = [ participant_1_power_values, participant_2_power_values]
            connectivity_value_list = [participant_1_connectivity_values, participant_2_connectivity_values]
            for participant in range(2):
                condition_names = ['Synchronous/Egalitarian', 'Synchronous/LeaderFollower', 'Synchronous/FollowerLeader', 'Complementary/Egalitarian', 'Complementary/LeaderFollower', 'Complementary/FollowerLeader','Individual/Egalitarian']
                participant_power_values = power_value_list[participant]
                participant_connectivity_values = connectivity_value_list[participant]

             
                for condition in condition_names:  

                    try:
                        # subselect all 
                       # print(condition)
                        filtered_dict = filter_dict_by_substrings(participant_power_values, [condition, 'Start'])
                        #print(filtered_dict.keys())
                        matrices = []
                        for key, matrix in filtered_dict.items():
                            if isinstance(matrix, np.ndarray):  # Ensure the value is a numpy array (matrix)
                            
                                matrices.append(matrix)
                            else:
                                raise ValueError(f"Value for key '{key}' is not a numpy array.")

                        # Stack matrices along a new dimension and compute the mean along that dimension
                        baseline_data = np.nanmean(np.stack(matrices), axis=0)

                        for trial in range(1,21):



                            try:
                                if 'Complementary' in condition:
                                    key = condition + '/Success/Desyncpoint/' + str(trial)
                                else:
                                    key = condition + '/Success/Checkpoint/' + str(trial)
                                
                                
                                trialmatrix = participant_power_values[key]
                                baselined_trialmatrix = trialmatrix 
                                power_trialmatrix = trialmatrix
                                baselined_trialmatrix = 10*np.log(trialmatrix / baseline_data)
                                #print(baseline_data)

                                # now get the frontal alpha power:

                                frontal_channels = ["AF3", "AFz", "AF4", "F3", "F1", "Fz", "F2", "F4"]
                                channel_numbers = get_channel_numbers(frontal_channels)
                                alpha_frontal = np.nanmean(baselined_trialmatrix[channel_numbers, 10-1])
                                

                                
                                # now get some connectivity measures (no baselining)
                                trialmatrix = participant_connectivity_values[key]


                                # also average over epochs: 

                                # Parietal-to-motor (asymmetric)
                                parietal_channels = ['CP6', 'P8', 'P6', 'P4', 'CP4', 'TP8']
                                channel_numbers = get_channel_numbers(parietal_channels)
                                alpha_parietal_motor = np.nanmean(trialmatrix[:,channel_numbers, 10-1, 1]) # 0 is parietal, 1 is motor
                                # Parietal-to-parietal (symmetric)
                                parietal_channels = ['CP6', 'P8', 'P6', 'P4', 'CP4', 'TP8']
                                channel_numbers = get_channel_numbers(parietal_channels)
                                alpha_parietal_parietal = np.nanmean(trialmatrix[:,channel_numbers, 10-1, 0])

                                # motor-to-motor (symmetric)
                                motor_channels = ["C3", "C1", "Cz", "C2", "C4"]
                                channel_numbers = get_channel_numbers(motor_channels)
                                alpha_motor_motor = np.nanmean(trialmatrix[:,channel_numbers, 10-1, 1])

                                print('save')
                                
                                df = df._append({'participant':participant+1, 'pair': pair, 'group': group, 'condition': condition, 'trial': trial, 'alpha_frontal': alpha_frontal, 'alpha_parietal_motor': alpha_parietal_motor, 'alpha_parietal_parietal': alpha_parietal_parietal, 'alpha_motor_motor': alpha_motor_motor}, ignore_index=True)

                            
                                # Perform operations that could fail
                            except KeyError as e:
                                print(f"KeyError: {e} for condition {condition} and trial {trial}")
                            except ValueError as e:
                                print(f"ValueError: {e} for key {key}")
                            except Exception as e:
                                print(f"Unexpected error occurred: {e} with size baselined_trialmatrix {np.shape(baseline_data)} and size trialmatrix {np.shape(power_trialmatrix)}")
                    # if condition was not found
                    except:
                            continue
            



                

storepath = os.path.join(path, 'trial_based_EEG_data.csv')
df.to_csv(storepath, index=False)           

     
                    
                  

                   

