######################
# Take the preprocessed data and do single-trial moement analysis
######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from Trial import TrialClass
import os
from scipy import signal
from Pair import PairClass
import scipy.io
import seaborn as sns
from scipy.fft import fft, fftfreq
import pickle
import re
from scipy.stats import mannwhitneyu
from scipy.signal import hilbert


path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\behavioral data"
#path = r"C:\Users\Administrator\Documents\Google\PhD documents\PhD documents\HYPERSCANNING_GAMEDATA"

os.chdir(path)

#how to exclude the big outlier positions at the beginning of trial??
# begin at first nonzero value of either one
# if over a threshold value: begin calculations at index after that threshold


#correlation of speed is good measure of synchronous movement (?) but not a general measure of jaggedness


# put the speeds in cm/seconds



with open(r"CivilianFiles.pickle", "rb") as input_file:
    CivilianList = pickle.load(input_file)
with open(r"MilitaryFiles.pickle", "rb") as input_file:
    MilitaryList = pickle.load(input_file)


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


def load_points(condition, level):
    if int(level) > 10:
        _Level = 'Level' + str(int(level)-10)
    else:
        _Level = 'Level' + str(level)
    obstacle_path = os.path.join(path,'LEVELDATA',condition,_Level)
    df = pd.read_csv(os.path.abspath(os.path.join(obstacle_path, 'CheckPointList_1.csv')), sep='delimiter', header=None, engine='python')
    Checkpos_x_1 = []
    Checkpos_y_1 = []
    for string in df[0]:
        Checkpos_x_1.append(string[1:].split(",")[0])
        Checkpos_y_1.append(string[1:].split(",")[1])
    df = pd.read_csv(os.path.abspath(os.path.join(obstacle_path, 'DesyncPointList_1.csv')), sep='delimiter', header=None, engine='python')
    Desyncpos_x_1 = []
    Desyncpos_y_1 = []
    for string in df[0]:
        Desyncpos_x_1.append(string[1:].split(",")[0])
        Desyncpos_y_1.append(string[1:].split(",")[1])

    ## player 2
    df = pd.read_csv(os.path.abspath(os.path.join(obstacle_path, 'CheckPointList_2.csv')), sep='delimiter', header=None, engine='python')
    Checkpos_x_2 = []
    Checkpos_y_2 = []
    for string in df[0]:
        Checkpos_x_2.append(string[1:].split(",")[0])
        Checkpos_y_2.append(string[1:].split(",")[1])
    df = pd.read_csv(os.path.abspath(os.path.join(obstacle_path, 'DesyncPointList_2.csv')), sep='delimiter', header=None, engine='python')
    Desyncpos_x_2 = []
    Desyncpos_y_2 = []
    for string in df[0]:
        Desyncpos_x_2.append(string[1:].split(",")[0])
        Desyncpos_y_2.append(string[1:].split(",")[1])

    if int(level) > 10:
        for pos in range(len(Checkpos_x_1)):
            Checkpos_x_1[pos] = - float(Checkpos_x_1[pos])
            Checkpos_x_2[pos] = - float(Checkpos_x_2[pos])

    if int(level) > 10:
        for pos in range(len(Desyncpos_x_1)):
            Desyncpos_x_1[pos] = - float(Desyncpos_x_1[pos])
            Desyncpos_x_2[pos] = - float(Desyncpos_x_2[pos])

    return Checkpos_x_1, Checkpos_y_1, Desyncpos_x_1, Desyncpos_y_1, Checkpos_x_2, Checkpos_y_2, Desyncpos_x_2, Desyncpos_y_2


def calculate_KOP(signal1, signal2):
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract phase
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    phase_matrix = np.vstack((phase1, phase2))

    KOP = np.abs(np.mean(np.exp(1j * phase_matrix), 0))
    return KOP


def calculate_phase_difference(signal1, signal2):
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract phase
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    phase2 - phase1

    return phase2 - phase1

def calculate_average_PLV(signal1, signal2, window_length, window_step):
    # calculate windowed PLV
    window_start = 0
    window_end = window_start + window_length
    simulation_length = len(signal1)
    plv_in_time = []
    interval_times = []

    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract phase
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    

    while (window_start + window_length) < simulation_length:
        
        interval_times.append(window_start + window_length/2)
        # Compute phase difference
        phase_difference = np.exp(1j * (phase2[window_start:window_end] - phase1[window_start:window_end]))

        # Compute Phase Locking Value (PLV)
        PLV = np.abs(np.mean(phase_difference))
        counter = 0
        window_start += window_step
        window_end += window_step
        counter += 1
        plv_in_time.append(PLV)
    


    mean_plv = np.mean(plv_in_time)

    return plv_in_time, interval_times, mean_plv


def calculate_average_KOP(signal1, signal2, window_length, window_step):
    # calculate windowed PLV
    window_start = 0
    window_end = window_start + window_length
    simulation_length = len(signal1)
    KOP_in_time = []
    interval_times = []

    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract phase
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    phase_matrix = np.vstack((phase1, phase2))



    while (window_start + window_length) < simulation_length:
        interval_times.append(window_start + window_length/2)
        # Compute phase difference
        KOP_raw = np.abs(np.mean(np.exp(1j * phase_matrix[:,window_start: window_end]), 0))

        KOP = np.mean(KOP_raw)
        counter = 0
        window_start += window_step
        window_end += window_step
        counter += 1
        KOP_in_time.append(KOP)

    mean_KOP = np.mean(KOP_in_time)

    return KOP_in_time, interval_times, mean_KOP


Mil_Sync_Egal_dV = []
Mil_Sync_LF_dV = []
Mil_Sync_Egal_dT = []
Mil_Sync_LF_dT = []

Civ_Sync_Egal_dV = []
Civ_Sync_LF_dV = []
Civ_Sync_Egal_dT = []
Civ_Sync_LF_dT = []


Mil_Completion_Times = []
Mil_Trial_Completion_Time = []


Civ_Completion_Times = []
Civ_Trial_Completion_Time = []

ListList = []
ListList.append(MilitaryList)
ListList.append(CivilianList)

Military_Completion_Times = []
Civilian_Completion_Times = []



print('start')
#condition = 'Sync_Egalitarian'# 'Sync_Egalitarian': #'Desync_LF':#'Sync_Egalitarian':#'Sync_Solo':
for Li in range(2):
    List = ListList[Li]
    for pair in List:
        Pair_dV = []
        Pair_dT = []
        print(pair.Pair)
        Pair_Completion_Times = []
        Pair_Trial_Completion_Time = []
      #  print(pair.Pair)
        for condition in ['Sync_LF']: #, 'Desync_Egalitarian', 'Sync_LF', 'Sync_FL']:
            tries = np.zeros((20,))
            times = np.zeros((20,))

            for trial in pair.TrialList:

                if trial.Success == False:
                    continue
                # print(trial.Success)
                # print(trial.CompletionTime)
                # continue

                # if tries[int(trial.TrialNumber)-1] > 1:
                #     continue


                if trial.Condition == condition and not int(trial.TrialNumber) == 4 and not int(trial.TrialNumber) == 14: #trial 4 causes unknown error
                    tries[int(trial.TrialNumber) - 1] += 1
                    times[int(trial.TrialNumber) - 1] += trial.CompletionTime
            #         if trial.Success == True or 1: #and tries[trial.TrialNumber] == 1:
                    if float(trial.time[-1]) < 2: #or int(trial.TrialNumber) != 5:
                        continue
                    
                    
                    # Create a single figure with three subplots
                    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

                    
                    Checkpos_x_1, Checkpos_y_1, Desyncpos_x_1, Desyncpos_y_1, Checkpos_x_2, Checkpos_y_2, Desyncpos_x_2, Desyncpos_y_2 = load_points(condition, trial.TrialNumber)
                    #go to the start of actual movement
                    startindex = 0
                    movement_start = False

                    grad_1_x = np.gradient(np.asarray(trial.Player_1_x))
                    grad_2_x = np.gradient(np.asarray(trial.Player_2_x))



                    grad_1_y = np.gradient(np.asarray(trial.Player_1_y))
                    grad_2_y = np.gradient(np.asarray(trial.Player_2_y))

                    speed_1 = Moving_average(np.sqrt(np.square(grad_1_x) + np.square(grad_1_y)),20)
                    speed_2 = Moving_average(np.sqrt(np.square(grad_2_x) + np.square(grad_2_y)),20)

                    speed_1 = speed_1[:-1]
                    speed_2 = speed_2[:-1]
                    trial.time = trial.time[:-1]

                    while movement_start == False:
                        startindex+=1
                        try:
                            if grad_1_x[startindex] != 0 or grad_2_x[startindex] != 0 or grad_1_y[startindex] != 0 or grad_1_y[startindex] != 0:
                                movement_start = True
                        except:
                            break
                    if movement_start == False:
                        continue

                    Threshold_passed = False
                    for index in range(0,len(grad_1_x)-1):
                        if np.abs(grad_1_x[index]) > 2 or np.abs(grad_2_x[index]) > 2 or np.abs(
                            grad_2_x[index]) > 2 or np.abs(grad_2_x[index]) > 2:
                            startindex = index + 10
                   # fig, ax = plt.subplots()

                    dTCheck = 0
                    dTDesync = 0
                    CheckCount = 0
                    DesyncCount = 0

                    for checkpoint in range(len(Checkpos_x_1)):
                        index_1 = 0
                        ok = True
                        while np.sqrt(
                            np.square(trial.Player_1_x[startindex:][index_1] - float(Checkpos_x_1[checkpoint])) + np.square(trial.Player_1_y[startindex:][index_1]- float(Checkpos_y_1[checkpoint]))) > 0.7:
                            index_1+=1
                            try:
                                trial.Player_1_x[startindex:][index_1+1]
                                trial.Player_2_x[startindex:][index_1+1]
                            except:
                                ok = False
                                break

                        if ok == True:
                            index_2 = 0
                            while np.sqrt(
                                    np.square(trial.Player_2_x[startindex:][index_2] - float(
                                        Checkpos_x_2[checkpoint])) + np.square(
                                        trial.Player_2_y[startindex:][index_2] - float(Checkpos_y_2[checkpoint]))) > 0.7:
                                index_2 += 1

                                try:
                                    trial.Player_1_x[startindex:][index_2]
                                    trial.Player_2_x[startindex:][index_2]
                                except:
                                    ok = False
                                    break

                            if ok == True:
                                CheckCount+=1
                                axs[0].axvspan(trial.time[startindex:][index_1], trial.time[startindex:][index_2], color='green', alpha=0.2)
                                dTCheck+= np.abs(index_2-index_1)


                    for checkpoint in range(len(Desyncpos_x_1)):
                        index_1 = 0
                        ok = True
                        while np.sqrt(
                            np.square(trial.Player_1_x[startindex:][index_1] - float(Desyncpos_x_1[checkpoint])) + np.square(trial.Player_1_y[startindex:][index_1]- float(Desyncpos_y_1[checkpoint]))) > 0.7:
                            index_1+=1

                            try:
                                trial.Player_1_x[startindex:][index_1]
                                trial.Player_2_x[startindex:][index_1]

                            except:
                                ok = False
                                break

                        if ok == True:
                            index_2 = 0
                            while np.sqrt(
                                    np.square(trial.Player_2_x[startindex:][index_2] - float(
                                        Desyncpos_x_2[checkpoint])) + np.square(
                                        trial.Player_2_y[startindex:][index_2] - float(Desyncpos_y_2[checkpoint]))) > 0.8:
                                index_2 += 1

                                try:
                                    trial.Player_2_x[startindex:][index_2]
                                    trial.Player_1_x[startindex:][index_2]
                                except:
                                    ok = False
                                    break

                            if ok == True:
                                if condition == "Sync_Egalitarian" or condition == "Sync_LF" or condition == "Sync_FL":
                                    axs[0].axvspan(trial.time[startindex:][index_1], trial.time[startindex:][index_2], color='green',
                                                alpha=0.2)
                                    dTCheck += np.abs(index_2 - index_1)
                                    CheckCount+=1
                                else:
                                    axs[0].axvspan(trial.time[startindex:][index_1], trial.time[startindex:][index_2], color='red', alpha=0.2)
                                    dTDesync += np.abs(index_2 - index_1)
                                    DesyncCount+=1

                    if condition == "Sync_Egalitarian" or condition == "Sync_LF" or condition == "Sync_FL":
                        try:
                            dTCheck = dTCheck / CheckCount
                        except:
                            dTCheck = np.nan
                    else:
                        try:
                            dTCheck = dTCheck / CheckCount
                        except:
                            dTCheck = np.nan
                        try:
                            dTDesync = dTDesync / DesyncCount
                        except:
                            dTDesync = np.nan



                    player1plot, = axs[0].plot(trial.time[startindex:],speed_1[startindex:])
                    player1plot.set_label('Player 1')
                    player2plot, = axs[0].plot(trial.time[startindex:],speed_2[startindex:])
                    player2plot.set_label('Player 2')
                    # ax.legend()


                    correlation_x = np.corrcoef(Moving_average(grad_1_x[startindex:], 20),Moving_average(grad_2_x[startindex:], 20))
                    correlation_y = np.corrcoef(Moving_average(grad_1_y[startindex:], 20),Moving_average(grad_2_y[startindex:], 20))
                    correlation = (correlation_x + correlation_y)/2
                    #correlation_array = np.correlate(trial.Player_1_x[startindex:],trial.Player_2_x[startindex:], mode = 'full')
                    correlation_array = np.correlate(Moving_average(grad_1_x[startindex:], 20),Moving_average(grad_2_x[startindex:], 20), mode = 'full')
                    k = (np.argmax(correlation_array) - 0.5*(len(correlation_array)-1))/100
                    average_acceleration = np.mean(Moving_average(grad_1_x[startindex:], 20))

                    dV = 0
                    for index in range(startindex + 20, len(speed_1)-20):
                        dV += np.abs(speed_1[index] - speed_2[index]) / np.abs(speed_1[index] + speed_2[index])
                    dV = (2 / (len(speed_1) - 1 - startindex)) * dV

                    average_speed = np.mean(np.abs(speed_1[startindex:] + speed_2[startindex:]))/2

                    axs[0].set_title('pair ' + str(pair.Pair) + ' dV ' + str(dV) +  ' corr ' + str(np.round(correlation[1,0],4)) + ' lag ' + str(k) + ' accel ' + str(np.round(average_acceleration, decimals = 2)))


                    # player1plot, = plt.plot(trial.time[startindex:],speed_1[startindex:] - speed_2[startindex:])
                   # plt.show()

                    



                    #plt.show()
                    analytic_signal1 = hilbert(speed_1[startindex:])
                    analytic_signal2 = hilbert(speed_2[startindex:])

                    """
                    # Extract phase
                    phase1 = np.angle(analytic_signal1)
                    phase2 = np.angle(analytic_signal2)
                    phase_diff = phase2 - phase1
                    axs[1].plot(trial.time[startindex+20:-20], phase1)
                    axs[1].plot(trial.time[startindex+20:-20], phase2)
                    #axs[1].plot(trial.time[startindex+20:-20], phase_diff)
                    #plt.show()

                    phase_matrix = np.vstack((phase1, phase2))
                    KOP_raw = np.abs(np.mean(np.exp(1j * phase_matrix), 0))
                    axs[2].plot(trial.time[startindex+20:-20], np.abs(phase2-phase1))#KOP_raw)
                    axs[2].plot(trial.time[startindex+20:-20], phase2 - phase1)

                    """
                    modified_time = trial.time[startindex:]
                    window_length = 100
                    window_step = 1
                    KOP_in_time, interval_times, mean_KOP = calculate_average_KOP(speed_1[startindex:], speed_2[startindex:], window_length, window_step)
                    axs[2].plot([modified_time[int(i)] for i in interval_times], KOP_in_time)

                    window_length = 20
                    KOP_in_time, interval_times, mean_KOP = calculate_average_KOP(speed_1[startindex:], speed_2[startindex:], window_length, window_step)
                    axs[1].plot([modified_time[int(i)] for i in interval_times], KOP_in_time)

                    modified_time = trial.time[startindex:]
                    window_length = 100
                    window_step = 1
                    PLV_in_time, interval_times, mean_plv = calculate_average_PLV(speed_1[startindex:], speed_2[startindex:], window_length, window_step)
                    axs[2].plot([modified_time[int(i)] for i in interval_times],PLV_in_time)
                    window_length = 20
                    PLV_in_time, interval_times, mean_plv = calculate_average_PLV(speed_1[startindex:], speed_2[startindex:], window_length, window_step)
                    axs[1].plot([modified_time[int(i)] for i in interval_times],PLV_in_time)
                    plt.show()

                    N = len(grad_1_x[startindex:])

                    yf = fft(speed_1[startindex:] - speed_2[startindex:])
                    #print(len(yf))
                    xf = fftfreq(len(speed_1[startindex:]- speed_2[startindex:]), 1 / 100)


                    # plt.plot(xf, np.abs(yf))
                    # plt.xlim([0, 10])
                  #  plt.show()

                    Pair_dV.append(dV)
                    Pair_dT.append(dTCheck)

            if len(Pair_dV) > 0 and len(Pair_dT) > 0:
                if Li == 0:
                    if condition == "Sync_Egalitarian":
                        Mil_Sync_Egal_dV.append(np.nanmean(Pair_dV))
                        Mil_Sync_Egal_dT.append(np.nanmean(Pair_dT))
                    elif condition == "Sync_LF" or condition == "Sync_FL":
                        Mil_Sync_LF_dV.append(np.nanmean(Pair_dV))
                        Mil_Sync_LF_dT.append(np.nanmean(Pair_dT))
                elif Li == 1:
                    if condition == "Sync_Egalitarian":
                        Civ_Sync_Egal_dV.append(np.nanmean(Pair_dV))
                        Civ_Sync_Egal_dT.append(np.nanmean(Pair_dT))
                    elif condition == "Sync_LF" or condition == "Sync_FL":
                        Civ_Sync_LF_dV.append(np.nanmean(Pair_dV))
                        Civ_Sync_LF_dT.append(np.nanmean(Pair_dT))
                    av_trial_times = times / tries
                    Pair_Completion_Times.append(times)


        if Li == 0:
            Mil_Completion_Times.append(np.nansum(Pair_Completion_Times))
            Mil_Trial_Completion_Time.append(np.nanmean(Pair_Completion_Times))
            print(Mil_Completion_Times)
        else:
            Civ_Completion_Times.append(np.nansum(Pair_Completion_Times))
            Civ_Trial_Completion_Time.append(np.nanmean(Pair_Completion_Times))


data = [Mil_Sync_Egal_dV, Mil_Sync_LF_dV,Civ_Sync_Egal_dV,Civ_Sync_LF_dV]
plt.figure()
plt.boxplot(data)
plt.show()

with open(r"dVData.pickle", "wb") as output_file:
    pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)

data = [Mil_Sync_Egal_dT, Mil_Sync_LF_dT,Civ_Sync_Egal_dT,Civ_Sync_LF_dT]
plt.figure()
plt.boxplot(data)
plt.show()

with open(r"dTData.pickle", "wb") as output_file:
    pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)

data = [Mil_Completion_Times, Civ_Completion_Times]
plt.figure()
plt.boxplot(data)
Mav = np.round(np.mean(Mil_Completion_Times))
Cav = np.round(np.mean(Civ_Completion_Times))
w, p = mannwhitneyu(Mil_Completion_Times, Civ_Completion_Times)
plt.title('Total time spent on all trials'+ ' M: ' + str(Mav) + ' C: ' + str(Cav) + ' p = ' + str(np.round(p,4 )))
plt.show()



data = [Mil_Trial_Completion_Time, Civ_Trial_Completion_Time]
plt.figure()
plt.boxplot(data)
Mav = np.round(np.mean(Mil_Trial_Completion_Time))
Cav = np.round(np.mean(Civ_Trial_Completion_Time))
w, p = mannwhitneyu(Mil_Trial_Completion_Time, Civ_Trial_Completion_Time)
plt.title('Average time spent on a trial (all tries)'+ ' M: ' + str(Mav) + ' C: ' + str(Cav) + ' p = ' + str(np.round(p,4 )))
plt.show()



# print(np.concatenate((np.array(Checkpos_x_1), np.array(Desyncpos_x_1))))
# firstpoint_x = np.min(np.concatenate((np.array(Checkpos_x_1), np.array(Desyncpos_x_1))).astype(np.float))
# lastpoint_x = np.max(np.concatenate((np.array(Checkpos_x_1), np.array(Desyncpos_x_1))).astype(np.float))
# first_index = 0
# last_index = 0
# print(firstpoint_x)
# print(lastpoint_x)
# index = startindex
# while index < len(trial.Player_1_x):
#     if trial.Player_1_x[index] > firstpoint_x or trial.Player_2_x[index] > firstpoint_x:
#         if first_index == 0:
#             first_index = index
#     if trial.Player_1_x[index] > lastpoint_x or trial.Player_2_x[index] > lastpoint_x:
#         if last_index == 0:
#             last_index = index
#     index += 1