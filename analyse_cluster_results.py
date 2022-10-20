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

# load in raw file to use channel locations for plotting
path = r"C:\Users\Administrator\Documents\MATLAB\Hyperscanning Analysis"

fig = plt.figure()
frequencies = ['theta', 'alpha', 'beta', 'gamma']


for frequency in range(4):
    storePath = os.path.join(path, "python_data",  "_sync_egal_clusters" + str(frequency) + ".pickle")
    #open cluster results
    with open(storePath , "rb") as input_file:
        cluster_matrix, pval= pickle.load(input_file)

    storePath = os.path.join(path, "python_data\plot_cluster_epochs_object.pickle")
    #open the plotting layout object
    with open(storePath , "rb") as input_file:
        epochs_object = pickle.load(input_file)

    print(cluster_matrix)
    print(cluster_matrix.shape)
    #epochs_object.info['chs'] = epochs_object.info['chs'][:64]
    #epochs_object.ch_names = epochs_object.ch_names[:64]
    fig.suptitle("DeSync")
    ax = fig.add_subplot(4, 1, frequency + 1, aspect = 1)
    ax.set_title(frequencies[frequency] + " (p = " + str(pval) + ")" )
    ax.axis("off")
    viz.plot_2d_topomap_inter(ax)
    viz.plot_sensors_2d_inter(epochs_object, epochs_object, lab = False) # bads are represented as squares
    viz.plot_links_2d_inter(epochs_object, epochs_object, C=cluster_matrix, threshold=2.5, steps=1)





plt.tight_layout()
plt.savefig(os.path.join(path, 'full_clusters.png'))
plt.show(block = True)

#viz.viz_2D_topomap_inter(epo1 = epochs_object, epo2 = epochs_object, C = cluster_matrix, threshold = 3, steps = 10)
#plt.show(block = True)

