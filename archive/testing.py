import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from Trial import TrialClass
import os
from scipy import signal
from behavioral.Pair import PairClass
import scipy
import seaborn as sns
import pickle
import re



path = r"C:\Users\Administrator\Desktop\HYPERSCANNING_GAMEDATA"
os.chdir(path)


with open(r"CivilianFiles.pickle", "rb") as input_file:
    CivilianList = pickle.load(input_file)
with open(r"MilitaryFiles.pickle", "rb") as input_file:
    MilitaryList = pickle.load(input_file)


military_succes_ratio = []
military_completion_time = []
for pair in MilitaryList:
    sum_succes = 0
    sum_total = 0
    trial_times = 0
    for i in range(len(pair.Trial_conditions)):
        if pair.Trial_conditions[i] != 5 and pair.Trial_conditions[i] != 0:
            sum_succes += pair.Trial_successes[i]
            sum_total += 1
            trial_times += pair.Trial_completion_times[i]
    successratio = sum_succes / sum_total
    average_completion_time = trial_times / sum_total
    # successratio = sum(pair.Trial_successes) / len(pair.Trial_successes)
    # average_completion_time = sum(np.asarray(pair.Trial_completion_times)[np.asarray(pair.Trial_successes)]) / len(
    #   np.asarray(pair.Trial_completion_times)[np.asarray(pair.Trial_successes)])

    military_succes_ratio.append(successratio)
    military_completion_time.append(average_completion_time)

civilian_succes_ratio = []
civilian_completion_time = []
for pair in CivilianList:
    sum_succes = 0
    sum_total = 0
    trial_times = 0
    for i in range(len(pair.Trial_conditions)):
        if pair.Trial_conditions[i] != 5 and pair.Trial_conditions[i] != 0:
            sum_succes += pair.Trial_successes[i]
            sum_total += 1
            trial_times+= pair.Trial_completion_times[i]
    successratio = sum_succes/sum_total
    average_completion_time = trial_times/sum_total
    # successratio = sum(pair.Trial_successes) / len(pair.Trial_successes)
   # average_completion_time = sum(np.asarray(pair.Trial_completion_times)[np.asarray(pair.Trial_successes)]) / len(
     #   np.asarray(pair.Trial_completion_times)[np.asarray(pair.Trial_successes)])

    civilian_succes_ratio.append(successratio)
    civilian_completion_time.append(average_completion_time)

print(military_succes_ratio)
print(civilian_succes_ratio)


print(scipy.stats.mannwhitneyu(np.asarray(military_succes_ratio),  np.asarray(civilian_succes_ratio), alternative = 'greater'))

print(scipy.stats.mannwhitneyu(np.asarray(military_completion_time),  np.asarray(civilian_completion_time), alternative = 'greater'))

data = [np.asarray(military_succes_ratio),  np.asarray(civilian_succes_ratio)]
fig1, ax1 = plt.subplots()
ax1.set_title('Ratio of successful trials')
ax1.boxplot(data, labels = ['Military','Civilian'])
plt.show()

data = [np.asarray(military_completion_time),  np.asarray(civilian_completion_time)]
fig2, ax2 = plt.subplots()
ax2.set_title('Average completion time for successful trials')
ax2.boxplot(data, labels = ['Military','Civilian'])
plt.show()