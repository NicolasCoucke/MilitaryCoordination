import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("C:/Users/Administrator/Documents/MATLAB/Hyperscanning Analysis/mne data/time_locked_full_averaged_connectivity_values_153.csv")

for ROI in df['ROI_combination'].unique():  # ["right_central-right-temporal"]:
    for freq in ['Theta']:
        for condition in ["Synchronous/Egalitarian", "Complementary/Egalitarian"]:
            condition_condition = df['condition'] == condition
            freq_condition = df['frequency'] == freq
            ROI_condition = df['ROI_combination'] == ROI

            store = np.empty((0,11))
            for i_pair in range(1,26):
                pair_condition = df['pair'] == i_pair

                filtered_df = df[pair_condition & condition_condition & freq_condition & ROI_condition]
                if len(filtered_df) > 0:
                    connectivity_string = re.sub(r'(\S)\s+(\S)', r'\1,\2', filtered_df['wPLI'].values[0])
                    print(connectivity_string)
                    connectivity_value = 0
                    exec('connectivity_value = np.array(' + connectivity_string + ')')
                    print(connectivity_value)
                    store = np.vstack((store, connectivity_value - connectivity_value[0]))
                    """
                    if condition == "Synchronous/Egalitarian":
                        plt.plot(connectivity_value - connectivity_value[0], color = 'blue', linewidth = 1, alpha = 0.5)
                    else:
                        plt.plot(connectivity_value - connectivity_value[0], color= 'orange', linewidth = 1, alpha = 0.5)
                    """
            if condition == "Synchronous/Egalitarian":
                plt.plot(np.nanmean(store, axis=0), linewidth = 5, color = 'blue', alpha = 0.5)
            else:
                plt.plot(np.nanmean(store, axis=0), linewidth=5, color='orange', alpha = 0.5)
        plt.axvline(x=5, color = 'black')
        plt.axhline(y=0, color = 'black')
        plt.title(freq + '|' + ROI)
        plt.show()








for condition in ["Synchronous/Egalitarian", "Complementary/Egalitarian"]:

    for freq in ['Theta', 'Alpha', 'Beta', 'Gamma']:
        for i_pair in range(1, 26):
            pair_condition = df['pair'] == i_pair
            condition_condition = df['condition'] == condition
            freq_condition = df['frequency'] == freq

            for ROI in df['ROI_combination'].unique():
                ROI_condition = df['ROI_combination'] == ROI
                filtered_df = df[pair_condition & condition_condition & freq_condition & ROI_condition]
                if len(filtered_df) > 0:
                    connectivity_string = re.sub(r'(\S)\s+(\S)', r'\1,\2', filtered_df['wPLI'].values[0])
                    print(connectivity_string)
                    connectivity_value = 0
                    exec('connectivity_value = np.array(' + connectivity_string + ')')
                    print(connectivity_value)
                    plt.plot(connectivity_value - connectivity_value[0])

        plt.title(freq + '|' + condition + '|' + str(i_pair))
        plt.show()



