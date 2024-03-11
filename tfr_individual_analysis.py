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
individual_stack = np.zeros((44,512))
sync_egal_stack = np.zeros((44,512))
leader_stack = np.zeros((44,512))
follower_stack = np.zeros((44,512))
complementary_sync_egal_stack = np.zeros((44,512))
complementary_leader_stack = np.zeros((44,512))
complementary_follower_stack = np.zeros((44,512))

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

            #print(participant_1_power_values)
            print(participant_2_power_values.keys())

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
                print(pair)

                #if len(participant_power_values.keys()) != 7:
                 #   continue

          
                print(np.shape(participant_power_values['Individual']))
                individual_stack = np.dstack((individual_stack, participant_power_values['Individual']))
                sync_egal_stack = np.dstack((sync_egal_stack , participant_power_values['Synchronous/Egalitarian']))
                leader_stack = np.dstack((leader_stack , participant_power_values['Synchronous/Leader']))
                follower_stack = np.dstack((follower_stack , participant_power_values['Synchronous/Follower']))
                complementary_sync_egal_stack = np.dstack((complementary_sync_egal_stack , participant_power_values['Complementary/Egalitarian']))
                complementary_leader_stack = np.dstack((complementary_leader_stack , participant_power_values['Complementary/Leader']))
                complementary_follower_stack = np.dstack((complementary_follower_stack , participant_power_values['Complementary/Follower']))