import os
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import re
from matplotlib.lines import Line2D
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib
import mne
from mne.stats import f_threshold_mway_rm, f_mway_rm, fdr_correction




def get_two_way_channel_clusters(data, factor_levels, n_replications, factors):
    """
    given a data array of dimensions (conditions x replications x times) this functions returns the significant clusters for the two main effects and an interaction effect
    other inputs of the function should be the factor levels (usually [2, 4])

    """

    significant_segments = []
    significant_pvals = []
    n_replications = np.size(data,1)

    F_obs_dict = dict()
    clusters_dict = dict()
    cluster_p_values_dict = dict()

    for i, effect in enumerate(["A", "B", "A:B"]):
        #print(effect)
        # we have to define a new stat function for every effect
        def stat_fun(*args):
            return f_mway_rm(
                np.swapaxes(args, 1, 0),
                # args,
                factor_levels=factor_levels,
                effects=effect,
                return_pvals=False,
            )[0]

        pthresh = 0.05
        f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effect, pthresh)

        tail = 1
        n_permutations = 1000

        F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
            np.swapaxes(data, 0, 1),
            stat_fun=stat_fun,
            threshold=f_thresh,
            tail=tail,
            n_permutations=n_permutations,
            buffer_size=None,
            out_type="mask",
        )

        
        F_obs_dict[factors[i]] = F_obs
        clusters_dict[factors[i]] = clusters
        cluster_p_values_dict[factors[i]] = cluster_p_values

    
    return  F_obs_dict, clusters_dict, cluster_p_values_dict


def plot_two_way_channel_clusters(F_obs_dict, clusters_dict, cluster_p_values_dict, factors, info, significance_level, axs):
    
    
    for ax, factor in zip(axs, factors):

         # Initialize a mask with False (indicating no significance)
        mask = np.zeros((64,), dtype=bool)  # data.shape[1] is n_channels
        
        F_obs = F_obs_dict[factor] 
        clusters = clusters_dict[factors]
        cluster_p_values = cluster_p_values_dict[factor]

        # Iterate through clusters and their p-values
        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < significance_level:
                print(cluster)
                # If significant, set those channel indices in the mask to True
                # Assuming clusters is a list of tuples where the first element is the cluster index array
                cluster_indices = cluster
                mask[cluster_indices] = True

        # Now you have a mask where True indicates channels part of significant clusters

        # Plotting the significant channels on a topomap
        # You might want to create a dummy data array for plotting purposes, since plot_topomap needs data values.
        # One simple approach is to use the mask itself as data, which will highlight significant areas.
        dummy_data = mask.astype(float)  # Convert boolean mask to float for plotting

        # Plot the topomap with significant areas highlighted
        mne.viz.plot_topomap(dummy_data, info, mask=mask, cmap='Reds', axes=ax, sensors=True, show=False)
    
    