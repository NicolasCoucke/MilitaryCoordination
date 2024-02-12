#combine behavioral and neural data
#also decide which trials to throw out (or not)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from Trial import TrialClass
import os
import pickle
from scipy import signal
from behavioral.Pair import PairClass
import scipy.io
import h5py
from behavioral.CorrelationValues import Calculate_Correlations, get_trial_processed_values

path = r"C:\Users\Administrator\Desktop\HYPERSCANNING_GAMEDATA"
os.chdir(path)


with open(r"CivilianFiles.pickle", "rb") as input_file:
    CivilianList = pickle.load(input_file)
with open(r"MilitaryFiles.pickle", "rb") as input_file:
    MilitaryList = pickle.load(input_file)


def Create_trial_matrix(pair):
    trialmatrix = np.zeros((0, 6))
    print("P" + str(pair.Pair))
    print(len(pair.TrialList))
    for trial in pair.TrialList:
        conditionNumber = 0
        if trial.Condition == 'Training':
            conditionNumber = 1
        elif trial.Condition == 'Sync_Egalitarian':
            conditionNumber = 2
        elif trial.Condition == 'Sync_LF':
            conditionNumber = 3
        elif trial.Condition == 'Sync_FL':
            conditionNumber = 4
        elif trial.Condition == 'Sync_Solo':
            conditionNumber = 5
        elif trial.Condition == 'Desync_Egalitarian':
            conditionNumber = 6
        elif trial.Condition == 'Desync_LF':
            conditionNumber = 7
        elif trial.Condition == 'Desync_FL':
            conditionNumber = 8

        correlation, lag = Calculate_Correlations(trial)
        dV, dT = get_trial_processed_values(trial)
        print("dt" + str(dT))
        trialmatrix = np.append(trialmatrix,
                  np.array([[int(conditionNumber), int(trial.TrialNumber), int(trial.Success == True), float(trial.CompletionTime), float(dV), float(dT)]]), axis = 0)
   # print(trialmatrix)
    return trialmatrix


    # export behavioral data to MATLAB
 #make a dictionary from the data

behavioral_military = dict([])
for pair in MilitaryList:
    print("M" + str(pair.Pair))
    behavioral_military["Pair" + str(pair.Pair)] = Create_trial_matrix(pair)

behavioral_civilian = dict([])
for pair in CivilianList:
    print("C" + str(pair.Pair))
    behavioral_civilian["Pair" + str(pair.Pair)] = Create_trial_matrix(pair)

scipy.io.savemat("military_trial_data_v3.mat", behavioral_military)
scipy.io.savemat("civilian_trial_data_v3.mat", behavioral_civilian)



