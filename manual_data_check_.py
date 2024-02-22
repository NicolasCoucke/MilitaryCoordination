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
import pickle
#print(sys.path)
sys.path.append('C:/Users/Administrateur/MilitaryCoordination/')



with open(r"data_pair_3.pickle","rb") as input_file:
    cleaned_epochs_AR = pickle.load(input_file)


mne.Epochs.plot(cleaned_epochs_AR[0], n_channels=64, block = True)

mne.Epochs.plot(cleaned_epochs_AR[1], n_channels=64, block = True)