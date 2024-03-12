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
from mne.time_frequency import tfr_morlet
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
# loop through all data files
individual_stack = np.zeros((44,512))
sync_egal_stack = np.zeros((44,512))
leader_stack = np.zeros((44,512))
follower_stack = np.zeros((44,512))
complementary_sync_egal_stack = np.zeros((44,512))
complementary_leader_stack = np.zeros((44,512))
complementary_follower_stack = np.zeros((44,512))

stack_dict = dict()
condition_names = ['Synchronous/Egalitarian', 'Synchronous/Leader', 'Synchronous/Follower', 'Individual', 'Complementary/Egalitarian', 'Complementary/Leader', 'Complementary/Follower']
for condition in condition_names:
    condition_stack = np.zeros((44,512))    
    stack_dict[condition] = condition_stack

biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
# change the channel names in our epochs so that they are the same as the montage
channels = biosemi64_montage.ch_names
# get motor channel ids:
motor_channels = ["C3", "C1", "Cz", "C2", "C4"]
motor_channel_numbers = motor_channels
for i in range(len(motor_channels)):
    ch_name = motor_channels[i]
    index = channels.index(ch_name)
    motor_channel_numbers[i] = index

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

            #if len(participant_1_power_values.keys()) != 7:
            #    continue

            for participant_power_values in [participant_1_power_values, participant_2_power_values]:
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

                skip = False
                for condition in condition_names:
                        if condition not in participant_power_values.keys():
                            skip = True
                if skip:
                    continue

                try:
                    

                    for condition in condition_names:
                        
                            condition_data = participant_power_values[condition]
                            
                            channel_data = np.mean(condition_data[motor_channel_numbers,:,:], axis = 0)
                            

                            stack_dict[condition] = np.dstack((stack_dict[condition], channel_data))
                except:
                    continue


# create_epoch_object
pair = 1


prep_path = os.path.join(path, "time locked preprocessed")
file_path = os.path.join(prep_path, "pair_1")
with open(file_path,"rb") as input_file:
    cleaned_epochs_AR = pickle.load(input_file)

preproc_S1 = cleaned_epochs_AR[0]


preproc_S1 = preproc_S1['Individual']

# Define frequencies of interest
freqs = np.arange(1, 45, 1)  # 1 to 40 Hz in 1 Hz steps
n_cycles = freqs / 4.  # Different number of cycle per frequency

# Define wavelet parameters
decim = 2  # To reduce computation time, you can increase this number
n_jobs = 1  # Number of parallel jobs to run. Can be increased if your machine supports it.

motor_channels = ["C3", "C1", "Cz", "C2", "C4"]
frontal_channels = []




power = tfr_morlet(preproc_S1, freqs=freqs, n_cycles=n_cycles, use_fft=True, picks = ['Cz'],
                    return_itc=False, decim=decim, n_jobs=n_jobs, average = True)    



sync_egal_averaged = np.nanmean(stack_dict['Synchronous/Leader'], axis = 2)
print(np.shape(sync_egal_averaged))
power.data  = sync_egal_averaged[np.newaxis,:,:]
power.plot(baseline=(-1, -1), mode='mean', title='MEG 0211 Power')



"""
 print(np.shape(participant_power_values['Individual']))
individual_stack = np.dstack((individual_stack, participant_power_values['Individual']))
sync_egal_stack = np.dstack((sync_egal_stack , participant_power_values['Synchronous/Egalitarian']))
leader_stack = np.dstack((leader_stack , participant_power_values['Synchronous/Leader']))
follower_stack = np.dstack((follower_stack , participant_power_values['Synchronous/Follower']))
complementary_sync_egal_stack = np.dstack((complementary_sync_egal_stack , participant_power_values['Complementary/Egalitarian']))
complementary_leader_stack = np.dstack((complementary_leader_stack , participant_power_values['Complementary/Leader']))
complementary_follower_stack = np.dstack((complementary_follower_stack , participant_power_values['Complementary/Follower']))
"""