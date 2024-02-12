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
path = r"C:\Users\Administrator\Desktop\HYPERSCANNING_GAMEDATA"
os.chdir(path)


with open(r"CivilianFiles.pickle", "rb") as input_file:
    CivilianList = pickle.load(input_file)
with open(r"MilitaryFiles.pickle", "rb") as input_file:
    MilitaryList = pickle.load(input_file)

# with open(r"pairProgressions.pickle", "rb") as input_file:
#     pairProgressions = pickle.load(input_file)
#
# plt.plot(range(len(pairProgressions['pair_alignments'])), np.asarray(pairProgressions['pair_alignments']))
# plt.show()

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def lagged_alignment(velocity_x_1, velocity_y_1, velocity_x_2, velocity_y_2):
    ###### CALCULATE HERDING ALIGNMENT#######
    alignment_matrix = np.zeros((11,len(velocity_x_1)))
    alignment_matrix[:] = np.nan
    for time in range(len(velocity_x_1)):
        if time < 10:  # edge cases (cf zero padding)
            lag_array = np.linspace(-5, 0, num=11)
            directional_correlation_function = np.zeros((11,))
        elif time > 10:
            lag_array = np.linspace(0, 5, num=11)
            directional_correlation_function = np.zeros((11,))
        else:
            lag_array = np.linspace(-5, 5, num=21)
            directional_correlation_function = np.zeros((21,))
        for lag in lag_array:
            t_vector = range(len(velocity_x_1))
            sample_x_1 = np.abs(velocity_x_1[time - int(lag)])

            sample_y_1 = np.abs(velocity_y_1[time - int(lag)])
            if sample_x_1 < 0.01:  # to prevent insigificant "pseudo-movements" to pollute the results
                sample_x_1 = 0
            if sample_y_1 < 0.01:
                sample_y_1 = 0
            velocity_vector_1 = np.asarray((sample_x_1, sample_y_1)).reshape(-1)

            sample_x_2 = np.abs(velocity_x_2[time - int(lag)])
            sample_y_2 = np.abs(velocity_y_2[time - int(lag)])
            if sample_x_2 < 0.01:  # to prevent insigificant "pseudo-movements" to pollute the results
                sample_x_2 = 0
            if sample_y_2 < 0.01:
                sample_y_2 = 0
            velocity_vector_2 = np.asarray((sample_x_2, sample_y_2)).reshape(-1)

            directional_correlation = np.inner(velocity_vector_1, velocity_vector_2) / (
                    np.linalg.norm(velocity_vector_1) * np.linalg.norm(velocity_vector_2))
            if np.isnan(directional_correlation):
                directional_correlation = 0
            alignment_matrix[int(lag+5),int(time)] =  directional_correlation
            #directional_correlation_function[int(lag)] = directional_correlation

    mean_alignments = np.nanmean(alignment_matrix, 1)
    general_mean_alignment = np.nanmean(alignment_matrix[6,:])
    max_index = np.nanargmax(mean_alignments)
    if(np.mean(mean_alignments[0:5]) > np.mean(mean_alignments[7:])):
        max_index = 0.1
    else:
        max_index = -0.1
    return  general_mean_alignment, max_index

def Moving_average(time_series, num_samples):
    filtered_series = time_series
    half_num = int(num_samples / 2)
    for i in range(0, len(time_series) - 1):
        if i < half_num:
            pre_samples = time_series[0:i]
        else:
            pre_samples = time_series[i - half_num:i]

        if i > len(time_series) - (half_num + 1):
            end_samples = time_series[i:len(time_series) - 1]
        else:
            # print(i + half_num)
            end_samples = time_series[i:i + half_num]

        sample = np.average(np.append(pre_samples, end_samples))
        filtered_series[i] = sample
    return filtered_series

    #mean_lagged_alignment = np.nanmean(directional_correlation_function)
def add_arrays(a,b):
    if len(a) < len(b):
        c = a.copy()
        c += a[:len(b)]
    else:
        c = a.copy()
        c[:len(b)] += b
    return c
#collective one

def get_graphs(participantList):

    SUM_pair_alignments = np.zeros((100,))
    SUM_pair_lags = np.zeros((100,))
    SUM_pair_cum_success = np.zeros((100,))
    SUM_pair_cum_time = np.zeros((100,))
    SUM_pair_cum_success[:] = np.nan
    SUM_pair_cum_time[:] = np.nan

    paircounter = 0
    for pair in participantList:
        #  print(pair.Pair)
        paircounter+=1
        trial_counter = 0
        success_counter = 0
        time_sum = 0
        pair_alignments = np.zeros((100,))
        pair_lags = np.zeros((100,))
        pair_cum_success = np.zeros((100,))
        pair_cum_time = np.zeros((100,))
        pair_cum_success[:] = np.nan
        pair_cum_time[:] = np.nan
        past_successes = []
        current_condition = "none"
        start_current_condition = 0
        for trial in pair.TrialList:
            print(trial.TrialNumber)
            if trial.Condition == 'Sync_Solo' or trial.Condition == 'Training':
                continue
            else:
                if current_condition == "none":
                    current_condition = trial.Condition
            print(current_condition)
            if current_condition != trial.Condition:
                break


            trial_counter+=1
            if trial_counter > 99:
                break
            if trial.Success:
                success_counter+=1
                time_sum += trial.CompletionTime
                pair_cum_time[trial_counter-1] = trial.CompletionTime
                past_successes.append(1)
            else:
                pair_cum_time[trial_counter-1] = np.nan
                past_successes.append(0)
            while len(past_successes) > 4:
                past_successes.pop(0)
            pair_cum_success[trial_counter-1] = np.mean(past_successes)
            #if(success_counter > 0):

               # pair_cum_time.append(time_sum/success_counter)
           # else:
               # pair_cum_time.append(0)


            current_condition = trial.Condition


        pair_cum_time = np.asarray(pair_cum_time)
        nans, x = nan_helper(pair_cum_time)
        print(pair_cum_time)
        print(nans)
        print(x)
        pair_cum_time[nans] = np.interp(x(nans), x(~nans), pair_cum_time[~nans])

      #  SUM_pair_cum_time = add_arrays(SUM_pair_cum_time, pair_cum_time)
       # SUM_pair_cum_success = add_arrays(SUM_pair_cum_success, np.asarray(pair_cum_success))

        SUM_pair_cum_time = np.vstack((SUM_pair_cum_time, pair_cum_time))
        SUM_pair_cum_success = np.vstack((SUM_pair_cum_success, np.asarray(pair_cum_success)))

    return SUM_pair_cum_time, SUM_pair_cum_success

mil_time, mil_success = get_graphs(MilitaryList)
civ_time, civ_success = get_graphs(CivilianList)
#plt.plot(range(100), pair_alignments/paircounter/np.asarray(pair_alignments))
#plt.plot(range(trial_counter),np.asarray(pair_lags))
plt.plot(range(40),np.nanmean(mil_time,0)[:40])
plt.plot(range(40),np.nanmean(civ_time,0)[:40])
plt.legend(['military', 'civilian'])
plt.xlabel('trial')
plt.ylabel('Completion time')
plt.show()

plt.plot(range(40),np.nanmean(mil_success,0)[:40])
plt.plot(range(40),np.nanmean(civ_success,0)[:40])
plt.legend(['military', 'civilian'])
plt.xlabel('trial')
plt.ylabel('Success ratio')
plt.show()


#for only one pair
for pair in CivilianList[20:]:
    #  print(pair.Pair)
    trial_counter = 0
    success_counter = 0
    time_sum = 0
    pair_alignments = []
    pair_lags = []
    pair_cum_success = []
    past_successes = []
    pair_cum_time = []
    current_condition = "none"
    start_current_condition = 0
    for trial in pair.TrialList:
        print(trial.TrialNumber)
        #
        # if int(trial.TrialNumber) == 5:
        #   #  print(trial.TrialNumber)
        #     if trial.Condition == 'Desync_LF':#'Sync_Egalitarian':#'Sync_Solo':
        #         if trial.Success == True:
        #go to the start of actual movement
        startindex = 0
        movement_start = False

        grad_1_x = np.gradient(np.asarray(trial.Player_1_x))
        grad_2_x = np.gradient(np.asarray(trial.Player_2_x))
        grad_1_y = np.gradient(np.asarray(trial.Player_1_y))
        grad_2_y = np.gradient(np.asarray(trial.Player_2_y))

        while movement_start == False:
            startindex+=1
            if grad_1_x[startindex] != 0 or grad_2_x[startindex] != 0 or grad_1_y[startindex] != 0 or grad_1_y[startindex] != 0:
                movement_start = True

        Threshold_passed = False
        for index in range(0,len(grad_1_x)-1):
            if np.abs(grad_1_x[index]) > 2 or np.abs(grad_2_x[index]) > 2 or np.abs(
                grad_2_x[index]) > 2 or np.abs(grad_2_x[index]) > 2:
                startindex = index + 10

        general_alignment, max_lag = lagged_alignment(Moving_average(grad_1_x[startindex:], 20),Moving_average(grad_2_x[startindex:], 20),Moving_average(grad_1_y[startindex:], 20),
                                    Moving_average(grad_2_y[startindex:], 20))
        pair_alignments.append(general_alignment)
        pair_lags.append(max_lag)

        trial_counter+=1
        if trial.Success:
            success_counter+=1
            time_sum += trial.CompletionTime
            pair_cum_time.append(trial.CompletionTime)
            past_successes.append(1)
        else:
            pair_cum_time.append(np.nan)
            past_successes.append(0)
        while len(past_successes) > 10:
            past_successes.pop(0)
        pair_cum_success.append(np.mean(past_successes))
        #if(success_counter > 0):

           # pair_cum_time.append(time_sum/success_counter)
       # else:
           # pair_cum_time.append(0)

        if(current_condition != trial.Condition or trial_counter == len(pair.TrialList)-2):
            _color = "white"
            if current_condition == "Sync_Egalitarian":
                _color = "limegreen"
            if current_condition == "Sync_LF":
                _color = "forestgreen"
            if current_condition == "Sync_FL":
                _color = "darkcyan"
            if current_condition == "Desync_Egalitarian":
                _color = "yellow"
            if current_condition == "Desync_LF":
                _color = "orange"
            if current_condition == "Desync_FL":
                _color = "darkkhaki"
            if current_condition == "Sync_Solo":
                _color = "blue"
            plt.axvspan(start_current_condition, trial_counter-1, color=_color,
                        alpha=0.4)
            start_current_condition = trial_counter
        current_condition = trial.Condition
        print(current_condition)

    pair_cum_time = np.asarray(pair_cum_time)
    nans, x = nan_helper(pair_cum_time)
    print(pair_cum_time)
    print(nans)
    print(x)
    pair_cum_time[nans] = np.interp(x(nans), x(~nans), pair_cum_time[~nans])

    pairProgressions = {}
    pairProgressions['pair_alignments'] = pair_alignments
    pairProgressions['pair_lags'] = pair_alignments
    pairProgressions['pair_cum_success'] = pair_alignments
    pairProgressions['pair_cum_time'] = pair_alignments
    with open(r"pairProgressions.pickle", "wb") as output_file:
        pickle.dump(pairProgressions, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")
    print(np.asarray(pair_alignments))
    plt.plot(range(len(pair_alignments)), np.asarray(pair_alignments)/np.nanmax(np.asarray(pair_alignments)))
    #plt.plot(range(trial_counter),np.asarray(pair_lags))
    plt.plot(range(trial_counter),np.asarray(pair_cum_success)/np.nanmax(np.asarray(pair_cum_success)))
    plt.plot(range(trial_counter),np.asarray(pair_cum_time)/np.nanmax(np.asarray(pair_cum_time)))
    plt.legend(['alignment','success rate', 'completion time'])
    plt.show()
