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
import pickle
import re


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


def Calculate_Correlations(trial):
    startindex = 0
    movement_start = False

    grad_1_x = np.gradient(np.asarray(trial.Player_1_x))
    grad_2_x = np.gradient(np.asarray(trial.Player_2_x))
    grad_1_y = np.gradient(np.asarray(trial.Player_1_y))
    grad_2_y = np.gradient(np.asarray(trial.Player_2_y))

    while movement_start == False:
        startindex += 1
        if grad_1_x[startindex] != 0 or grad_2_x[startindex] != 0 or grad_1_y[startindex] != 0 or grad_1_y[
            startindex] != 0:
            movement_start = True

        if startindex + 2 > len(grad_1_x): #if they don't move the whole trial
            movement_start = True

    Threshold_passed = False
    for index in range(0, len(grad_1_x) - 1):
        if np.abs(grad_1_x[index]) > 2 or np.abs(grad_2_x[index]) > 2 or np.abs(
                grad_2_x[index]) > 2 or np.abs(grad_2_x[index]) > 2:
            startindex = index + 10

    correlation_x = np.corrcoef(Moving_average(grad_1_x[startindex:], 20),
                                Moving_average(grad_2_x[startindex:], 20))
    correlation_y = np.corrcoef(Moving_average(grad_1_y[startindex:], 20),
                                Moving_average(grad_2_y[startindex:], 20))
    correlation = (correlation_x + correlation_y) / 2
    # correlation_array = np.correlate(trial.Player_1_x[startindex:],trial.Player_2_x[startindex:], mode = 'full')


    if len(Moving_average(grad_1_x[startindex:], 20)) != 0:
        correlation_array = np.correlate(Moving_average(grad_1_x[startindex:], 20),
                                         Moving_average(grad_2_x[startindex:], 20), mode='full')
        corr_value = np.round(correlation[1, 0], 4)
        k = (np.argmax(correlation_array) - 0.5 * (len(correlation_array) - 1)) / 100
    else:
        corr_value =  float("NaN")
        k =  float("NaN")


    return corr_value, k


path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\behavioral data"
os.chdir(path)

with open(r"CivilianFiles.pickle", "rb") as input_file:
    CivilianList = pickle.load(input_file)
with open(r"MilitaryFiles.pickle", "rb") as input_file:
    MilitaryList = pickle.load(input_file)


# def Moving_average(time_series, num_samples):
#     filtered_series = time_series
#     half_num = int(num_samples / 2)
#     for i in range(0, len(time_series) - 1):
#         if i < half_num:
#             pre_samples = time_series[0:i]
#         else:
#             pre_samples = time_series[i - half_num:i]
#
#         if i > len(time_series) - (half_num + 1):
#             end_samples = time_series[i:len(time_series) - 1]
#         else:
#            # print(i + half_num)
#             end_samples = time_series[i:i + half_num]
#
#         sample = np.average(np.append(pre_samples, end_samples))
#         filtered_series[i] = sample
#     return filtered_series


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

def get_trial_processed_values(trial):
    if not int(trial.TrialNumber) == 4 and not int(
            trial.TrialNumber) == 14:  # trial 4 causes unknown error
        if float(trial.time[-1]) < 2:  # or int(trial.TrialNumber) != 5:
            return np.nan, np.nan
        condition = trial.Condition
        Checkpos_x_1, Checkpos_y_1, Desyncpos_x_1, Desyncpos_y_1, Checkpos_x_2, Checkpos_y_2, Desyncpos_x_2, Desyncpos_y_2 = load_points(
            trial.Condition, trial.TrialNumber)
        # go to the start of actual movement
        startindex = 0
        movement_start = False

        grad_1_x = np.gradient(np.asarray(trial.Player_1_x))
        grad_2_x = np.gradient(np.asarray(trial.Player_2_x))

        grad_1_y = np.gradient(np.asarray(trial.Player_1_y))
        grad_2_y = np.gradient(np.asarray(trial.Player_2_y))

        speed_1 = Moving_average(np.sqrt(np.square(grad_1_x) + np.square(grad_1_y)), 20)
        speed_2 = Moving_average(np.sqrt(np.square(grad_2_x) + np.square(grad_2_y)), 20)

        while movement_start == False:
            startindex += 1
            try:
                if grad_1_x[startindex] != 0 or grad_2_x[startindex] != 0 or grad_1_y[startindex] != 0 or \
                        grad_1_y[startindex] != 0:
                    movement_start = True
            except:
                break
        if movement_start == False:
            return np.nan, np.nan

        Threshold_passed = False
        for index in range(0, len(grad_1_x) - 1):
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
                    np.square(trial.Player_1_x[startindex:][index_1] - float(
                        Checkpos_x_1[checkpoint])) + np.square(
                        trial.Player_1_y[startindex:][index_1] - float(Checkpos_y_1[checkpoint]))) > 0.7:
                index_1 += 1
                try:
                    trial.Player_1_x[startindex:][index_1 + 1]
                    trial.Player_2_x[startindex:][index_1 + 1]
                except:
                    ok = False
                    break

            if ok == True:
                index_2 = 0
                while np.sqrt(
                        np.square(trial.Player_2_x[startindex:][index_2] - float(
                            Checkpos_x_2[checkpoint])) + np.square(
                            trial.Player_2_y[startindex:][index_2] - float(
                                Checkpos_y_2[checkpoint]))) > 0.7:
                    index_2 += 1

                    try:
                        trial.Player_1_x[startindex:][index_2]
                        trial.Player_2_x[startindex:][index_2]
                    except:
                        ok = False
                        break

                if ok == True:
                    CheckCount += 1
                    plt.axvspan(trial.time[startindex:][index_1], trial.time[startindex:][index_2],
                                color='green', alpha=0.2)
                    dTCheck += np.abs(index_2 - index_1)

        for checkpoint in range(len(Desyncpos_x_1)):
            index_1 = 0
            ok = True
            while np.sqrt(
                    np.square(trial.Player_1_x[startindex:][index_1] - float(
                        Desyncpos_x_1[checkpoint])) + np.square(
                        trial.Player_1_y[startindex:][index_1] - float(Desyncpos_y_1[checkpoint]))) > 0.7:
                index_1 += 1

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
                            trial.Player_2_y[startindex:][index_2] - float(
                                Desyncpos_y_2[checkpoint]))) > 0.8:
                    index_2 += 1

                    try:
                        trial.Player_2_x[startindex:][index_2]
                        trial.Player_1_x[startindex:][index_2]
                    except:
                        ok = False
                        break

                if ok == True:
                    if condition == "Sync_Egalitarian" or condition == "Sync_LF" or condition == "Sync_FL":
                        plt.axvspan(trial.time[startindex:][index_1], trial.time[startindex:][index_2],
                                    color='green',
                                    alpha=0.2)
                        dTCheck += np.abs(index_2 - index_1)
                        CheckCount += 1
                    else:
                        plt.axvspan(trial.time[startindex:][index_1], trial.time[startindex:][index_2],
                                    color='red', alpha=0.2)
                        dTDesync += np.abs(index_2 - index_1)
                        DesyncCount += 1

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


        dV = 0
        for index in range(startindex + 50, len(speed_1) - 50):
            dV += np.abs(speed_1[index] - speed_2[index]) / np.abs(speed_1[index] + speed_2[index])
        dV = (2 / (len(speed_1) - 1 - startindex)) * dV

        average_speed = np.mean(np.abs(speed_1[startindex:] + speed_2[startindex:])) / 2
        return dV, dTCheck
    return np.nan, np.nan

        # plt.title('pair ' + str(pair.Pair) + ' dV ' + str(dV) + ' corr ' + str(
        #     np.round(correlation[1, 0], 4)) + ' lag ' + str(k) + ' accel ' + str(
        #     np.round(average_acceleration, decimals=2)))

        # player1plot, = plt.plot(trial.time[startindex:],speed_1[startindex:] - speed_2[startindex:])
        # plt.show()
        #
        # N = len(grad_1_x[startindex:])
        #
        # yf = fft(speed_1[startindex:] - speed_2[startindex:])
        # # print(len(yf))
        # xf = fftfreq(len(speed_1[startindex:] - speed_2[startindex:]), 1 / 100)

        # plt.plot(xf, np.abs(yf))
        # plt.xlim([0, 10])
        #  plt.show()