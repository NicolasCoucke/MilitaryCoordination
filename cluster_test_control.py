#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : cluster_test_population.py
# description     : Import and preprocess data
# author          : Nicolas Coucke
# date            : 2022-07-05
# version         : 1
# usage           : python cluster_test_population.py
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



alone_array = np.zeros((1,64*64,4))
sync_array = np.zeros((1,64*64,4))
#get the averaged wPLI values from the MATLAB matrices
raw_path = r"C:\Users\Administrator\Documents\MATLAB\Hyperscanning Analysis\python_data"



for pair in range(1,45):
    # get alone data
    filename = "pair_" + str(pair) + "_full_average_matrix_alone.mat"
    filepath = os.path.join(raw_path, filename)
    try:
        mat_contents = sio.loadmat(filepath)
    except:
        print("pair " + str(pair) + "not found")
    pair_data = np.asarray(mat_contents['averaged_data'])
    stretched_pair_data = stretch_coupling_matrix(pair_data)
    stretched_pair_data = np.expand_dims(stretched_pair_data, 0)
    alone_array = np.vstack((alone_array, stretched_pair_data))

    # get synchrony data
    filename = "pair_" + str(pair) + "_full_average_matrix_sync_egal.mat"
    filepath = os.path.join(raw_path, filename)
    try:
        mat_contents = sio.loadmat(filepath)
    except:
        print("pair " + str(pair) + "not found")
    pair_data = np.asarray(mat_contents['averaged_data'])
    stretched_pair_data = stretch_coupling_matrix(pair_data)
    stretched_pair_data = np.expand_dims(stretched_pair_data, 0)
    sync_array = np.vstack((sync_array, stretched_pair_data))

    

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
#metaconn_freq = np.tile(metaconn, (4,4))
#ch_con_freq = scipy.sparse.csr_matrix(metaconn_freq)
ch_con_freq = scipy.sparse.csr_matrix(metaconn)
#ch_con_freq = metaconn
for i in range(np.size(metaconn,0)):
    for j in range(np.size(metaconn,1)):
        if i == j:
            metaconn[i, j] = 0

#plt.imshow(metaconn)
#plt.show(block = True)

cluster_channel_dict = dict()
frequencies = ['theta', 'alpha', 'beta', 'gamma']

for frequency in range(4):
    DATA = [sync_array[1:,:,frequency], alone_array[1:,:,frequency]]
    data = DATA


    n_permutations = 5000
    tail = 0
    alpha = 0.05
    def stat_fun(*arg):
        return(scipy.stats.ttest_rel(arg[0], arg[1])[0])
    Stat_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(DATA,
                                                                                    stat_fun=stat_fun,
                                                                                    tail=tail,
                                                                                    n_permutations=n_permutations,
                                                                                    adjacency= ch_con_freq,
                                                                                    max_step = 2,
                                                                                    t_power=1,
                                                                                    threshold = 2.7,
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

    cluster_channels = np.array([0, 0])
    #convert cluster data to matrix data
    for connection in range(64*64):
        channel_1 = int(np.floor(connection/64))
        channel_2 = int(connection % 64)
        F_matrix[channel_1, channel_2] = Stat_obs[connection]#int(cluster_array[connection,3])
        cluster_matrix[channel_1, channel_2] = int(cluster_array[connection])
        

        if cluster_array[connection] == True:
            F_matrix_corrected[channel_1, channel_2] = Stat_obs[connection]
            cluster_channels = np.vstack((cluster_channels, np.array([channel_1+1, channel_2+1])))
        else:
            F_matrix_corrected[channel_1, channel_2] = 0
   # plt.imshow(F_matrix_corrected, cmap='hot', interpolation='nearest')
   # plt.show(block=True)

    cluster_channel_dict[frequencies[frequency]] = cluster_channels


    #save result of permutation analysis
    storePath = os.path.join(raw_path, "sync_clusters" + str(frequency) + ".pickle")
    with open(storePath, "wb") as output_file: 
        pickle.dump([F_matrix_corrected, np.min(cluster_p_values)], output_file, protocol=pickle.HIGHEST_PROTOCOL)


sio.savemat(os.path.join(raw_path, 'cluster_channels_sync.mat'), cluster_channel_dict)

plt.imshow(F_matrix, cmap='hot', interpolation='nearest')
plt.show(block=True)

plt.imshow(cluster_matrix, cmap='hot', interpolation='nearest')
plt.show(block=True)




