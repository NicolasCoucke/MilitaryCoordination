#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : cluster_test_aggregate.py
# description     : Import and preprocess data
# author          : Nicolas Coucke
# date            : 2022-07-05
# version         : 1
# usage           : python cluster_test_aggregate.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.13
# ==============================================================================

import io
from copy import copy
from collections import OrderedDict
#from hamcrest import none
import matplotlib.pyplot as plt
import numpy as np
import scipy
import h5py
import mne
import hypyp
import requests
import os
import pickle
import scipy.io as sio

from hypyp import (
    prep,
)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import stats
from hypyp import viz

def stretch_coupling_matrix(coupling_matrix):
    stretched_coupling_matrix = np.zeros((64*64,4))
    for freq in range(4):
        for channel_1 in range(64):
            for channel_2 in range(64):
                stretched_coupling_matrix[64*channel_1+channel_2,freq] = coupling_matrix[channel_1, channel_2, freq]

    return stretched_coupling_matrix



civilian_array = np.zeros((1,64*64,4))
military_array = np.zeros((1,64*64,4))
#get the averaged wPLI values from the MATLAB matrices
raw_path = r"C:\Users\Administrator\Documents\MATLAB\Hyperscanning Analysis\python_data"

for pair in range(1,45):
    filename = "pair_" + str(pair) + "_average_matrix.mat"
    filepath = os.path.join(raw_path, filename)
    try:
        mat_contents = sio.loadmat(filepath)
    except:
        print("pair " + str(pair) + "not found")
    pair_data = np.asarray(mat_contents['averaged_data'])
    stretched_pair_data = stretch_coupling_matrix(pair_data)
   # for connection in range(64*64):
       # print(stretched_pair_data[connection,:])
    stretched_pair_data = np.expand_dims(stretched_pair_data, 0)
    if pair < 20:
        civilian_array = np.vstack((civilian_array, stretched_pair_data))
    else:
        military_array = np.vstack((military_array, stretched_pair_data))

print(civilian_array[1,:75,2])
DATA = [civilian_array[1:,:,1], military_array[1:,:,1]]

# create a tuple of streched data 
electrodes = []
for channel_1 in range(64):
    for channel_2 in range(64):
        electrodes.append((channel_1, channel_2))

print(len(electrodes))
# get the adjacency matrix created in MATLAB
filepath = os.path.join(raw_path, "single_brain_adjacency_matrix.mat")
mat_contents = sio.loadmat(filepath)
single_brain_adjacency = np.asarray(mat_contents['neighbourhood_matrix'])
print(single_brain_adjacency.shape)

print('get adjacency')
# get the inter-brain adjacency matrix
#freq_list = [1]
#con_matrixTuple = stats.metaconn_matrix_2brains(electrodes, scipy.sparse.csr_matrix(single_brain_adjacency), freq_list, True) 

storePath = os.path.join(raw_path, "two_brain_adjacency.pickle")
#with open(storePath, "wb") as output_file: 
 #   pickle.dump(con_matrixTuple.metaconn, output_file, protocol=pickle.HIGHEST_PROTOCOL)
#print("done")
#print(con_matrixTuple.metaconn)


#open matrix
with open(storePath , "rb") as input_file:
    metaconn = pickle.load(input_file)

#make matrix work for the 4 frequencies
metaconn_freq = np.zeros((4*64*64, 4*64*64))
metaconn_freq[:64*64, :64*64] = metaconn

#import in the statscluster function

#Stat_obs, clusters, cluster_p_values, H0 = stats.statscluster(data = DATA, test = 'f oneway', factor_level = 2, tail = 0, ch_con_freq = scipy.sparse.csr_matrix(metaconn_freq), n_permutations = 100, alpha = 0.05)
data = DATA
test = 'f oneway'
factor_levels = 2
if test == 'f multipleway':
    if max(factor_level) > 2:
        correction = True
    else:
        correction = False
n_permutations = 1000
#metaconn_freq = np.tile(metaconn, (4,4))
#ch_con_freq = scipy.sparse.csr_matrix(metaconn_freq)
ch_con_freq = scipy.sparse.csr_matrix(metaconn)

tail = 0
alpha = 0.05
#def stat_fun(*arg):
   # return(scipy.stats.f_oneway(arg[0], arg[1])[0])

def stat_fun(*arg):
            return(scipy.stats.ttest_ind(arg[0], arg[1], equal_var=False)[0])

Stat_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(DATA,
                                                                                stat_fun=stat_fun,
                                                                                tail=tail,
                                                                                n_permutations=n_permutations,
                                                                                adjacency= ch_con_freq,
                                                                                t_power=1,
                                                                                threshold = 2,
                                                                                out_type='mask')
print(Stat_obs.shape)
print(len(clusters))
print((clusters[0].shape))
print(cluster_p_values)
print(H0.shape)



cluster_matrix = np.zeros((64,64))
F_matrix = np.zeros((64,64))
F_matrix_corrected = np.zeros((64,64))
cluster_array = clusters[np.argmin(cluster_p_values)]
#convert cluster data to matrix data
for connection in range(64*64):
    channel_1 = int(np.floor(connection/64))
    channel_2 = int(connection % 64)
    F_matrix[channel_1, channel_2] = Stat_obs[connection]#int(cluster_array[connection,3])
    cluster_matrix[channel_1, channel_2] = int(cluster_array[connection])
    
    if cluster_array[connection] == True:
        F_matrix_corrected[channel_1, channel_2] = Stat_obs[connection]
    else:
        F_matrix_corrected[channel_1, channel_2] = 0

plt.imshow(F_matrix_corrected, cmap='hot', interpolation='nearest')
plt.show(block=True)


plt.imshow(cluster_matrix, cmap='hot', interpolation='nearest')
plt.show(block=True)


#save result of permutation analysis
storePath = os.path.join(raw_path, "sync_egal_clusters.pickle")
with open(storePath, "wb") as output_file: 
   pickle.dump(F_matrix_corrected, output_file, protocol=pickle.HIGHEST_PROTOCOL)

