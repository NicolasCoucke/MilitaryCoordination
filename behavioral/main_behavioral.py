import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from Trial import TrialClass
import os
from scipy import signal
from behavioral.Pair import PairClass
import scipy.io
import seaborn as sns
import pickle
import re

#DATA STRUCTURE:
#Pair: extract all data from the pair
#Condition
#Trial
#


MilitaryList = []
CivilianList = []
ConditionList = []
TrialList = []
FailureList = []
trial = 0

isMilitary = True


path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\behavioral data"
os.chdir(path)
i=0
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".csv")):
            i+=1
            newpath = os.path.join(root, name)
            df = pd.read_csv(os.path.abspath(os.path.join(root, name)), sep='delimiter', header=None, engine='python')

            print(os.path.abspath(os.path.join(root, name)))
            data = df[0]
            pairnumber = int(re.search(r'\d+', name.split("_")[0]).group())
            pair = PairClass(pairnumber)
            pair.ParsePair(data)
            if "Military" in newpath:
                MilitaryList.append(pair)
            else:
                CivilianList.append(pair)


Obstacle_locations = dict([])
path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\behavioral data\LEVELDATA"
os.chdir(path)
i=0
for dirs in os.walk(path):
    Obstacle_locations[dirs] = dict([])
    for name in files:
        if name.endswith((".csv")):
            i+=1
            newpath = os.path.join(root, name)
            df = pd.read_csv(os.path.abspath(os.path.join(root, name)), sep='delimiter', header=None, engine='python')

            print(os.path.abspath(os.path.join(root, name)))
            data = df[0]
            pairnumber = int(re.search(r'\d+', name.split("_")[0]).group())
            pair = PairClass(pairnumber)
            pair.ParsePair(data)
            if "Military" in newpath:
                MilitaryList.append(pair)
            else:
                CivilianList.append(pair)






military_succes_ratio = []
military_completion_time = []
for pair in MilitaryList:
    successratio = sum(pair.Trial_successes)/len(pair.Trial_successes)
    average_completion_time = sum(np.asarray(pair.Trial_completion_times)[np.asarray(pair.Trial_successes)])/len(np.asarray(pair.Trial_completion_times)[np.asarray(pair.Trial_successes)])

    military_succes_ratio.append(successratio)
    military_completion_time.append(average_completion_time)

civilian_succes_ratio = []
civilian_completion_time = []
for pair in CivilianList:
    successratio = sum(pair.Trial_successes) / len(pair.Trial_successes)
    average_completion_time = sum(np.asarray(pair.Trial_completion_times)[np.asarray(pair.Trial_successes)]) / len(
        np.asarray(pair.Trial_completion_times)[np.asarray(pair.Trial_successes)])

    civilian_succes_ratio.append(successratio)
    civilian_completion_time.append(average_completion_time)

print(military_succes_ratio)
print(civilian_succes_ratio)

with open(r"MilitaryFiles.pickle", "wb") as output_file:
    pickle.dump(MilitaryList, output_file, protocol=pickle.HIGHEST_PROTOCOL)
print("done")

with open(r"CivilianFiles.pickle", "wb") as output_file:
    pickle.dump(CivilianList, output_file, protocol=pickle.HIGHEST_PROTOCOL)
print("done")







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



ax = sns.boxplot(x="population", y="success", hue="time",
                 data=[np.asarray(military_succes_ratio), np.asarray(civilian_succes_ratio)], linewidth=2.5)
plt.plot(ax)
plt.show()

ax = sns.boxplot(x="population", y="completion time", hue="time",
                 data=[np.asarray(military_completion_time), np.asarray(civilian_completion_time)], linewidth=2.5)


plt.plot(ax)
plt.show()







path = r"C:\Users\Administrator\Documents\MATLAB\Hyperscanning Analysis"
cohspctr = scipy.io.loadmat('P1P2_15_02_cohspectrm.mat')
print(cohspctr)

#with open(r"session5.pickle", "rb") as input_file:
    #  session = pickle.load(input_file)


















































def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))
save_maxlags = []
save_assymmetry = []
for k in range (1,len(TrialList)):
    for i in range(0,len(TrialList[k].Player_1_y)):
        TrialList[k].Player_1_y[i] -= 2.5
    for i in range(0,len(TrialList[k].Player_2_y)):
        TrialList[k].Player_2_y[i] += 2.5
    print(k)
    #i=500
    i = 0
    while (TrialList[k].Time[i] < 1):
        i += 1

    line1, = plt.plot(TrialList[k].Player_1_x[i:], TrialList[k].Time[i:])
    line2, = plt.plot(TrialList[k].Player_2_x[i:], TrialList[k].Time[i:])
    # plt.plot(TrialList[k].Player_1_x[i:], TrialList[k].Player_1_y[i:])
    # plt.plot(TrialList[k].Player_2_x[i:], TrialList[k].Player_2_y[i:])
    # plt.legend((line1, line2), ('P1', 'P2'))
    plt.title(TrialList[k].Condition)
    plt.show()
    print("postshow")


    plt.show()
