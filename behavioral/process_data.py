#########################
# Take the preprocessed data objects created with main_behavioral.py
# and create:
# 1) a data object with individual phases that can be linked to the EEG (at trial level - not yet evoked)
# 2) a data file with behavioral measures that can be used for analysis in R

# note: the data file with behavioral measures will created in two ways:
# - once based on a moving window averaged across the complete trials
# - once based on a 'evoked' segments that are extracted from crossing the checkpoints
# note to self: make sure to alsways take into account introduced lags
#########################

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert

# 1) data object for linking with EEG needs:
# (only successful trials)
# time series of the two (unlagged) phase time series
# timestamps of when the checkpoint is crossed by each participant
# beginning and end of the trial
# order of the participants doing each condition



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

print('testings')
path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\behavioral data"
#path = r"C:\Users\Administrator\Documents\Google\PhD documents\PhD documents\HYPERSCANNING_GAMEDATA"
os.chdir(path)




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



# open the parsed data
with open(r"CivilianFiles.pickle", "rb") as input_file:
    CivilianList = pickle.load(input_file)
with open(r"MilitaryFiles.pickle", "rb") as input_file:
    MilitaryList = pickle.load(input_file)


def distance(player_position, checkpoint_position):
    return np.sqrt( np.square(player_position[0] - checkpoint_position[0]) + np.square(player_position[1] - checkpoint_position[1]))


def get_distance_trajectory(trial, index, checkpoint_1, checkpoint_2):
    # get time locked trajectories if player two is first
    trajectory_1 = []
    trajectory_2 = []
    for traj_index in range(index - 100, index + 50):
        try:
            distance_1 = distance(
                [trial.Player_1_x[traj_index], trial.Player_1_y[traj_index]],
               checkpoint_1)
            distance_2 = distance(
                [trial.Player_2_x[traj_index], trial.Player_2_y[traj_index]],
                checkpoint_2)
            trajectory_1.append(distance_1)
            trajectory_2.append(distance_2)
        except:
            trajectory_1.append(np.nan)
            trajectory_2.append(np.nan)
    return trajectory_1, trajectory_2




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


def get_phase_measures(PLV, KOP, index):
    # get time locked trajectories if player two is first
    PLV_trajectory = []
    KOP_trajectory = []

    # compensate for the fact that plv and kop have one window length lag
    #index = index - 20 (not if we do not use the windowed plv etc)

    for traj_index in range(index - 100, index + 50):
        try:
            PLV_trajectory.append(PLV[traj_index])
            KOP_trajectory.append(KOP[traj_index])
        except:
            PLV_trajectory.append(np.nan)
            KOP_trajectory.append(np.nan)
    return PLV_trajectory, KOP_trajectory


def get_distance_trajectory(trial, index, checkpoint_1, checkpoint_2):
    # get time locked trajectories if player two is first
    trajectory_1 = []
    trajectory_2 = []
    for traj_index in range(index - 100, index + 50):
        try:
            distance_1 = distance(
                [trial.Player_1_x[traj_index], trial.Player_1_y[traj_index]],
               checkpoint_1)
            distance_2 = distance(
                [trial.Player_2_x[traj_index], trial.Player_2_y[traj_index]],
                checkpoint_2)
            trajectory_1.append(distance_1)
            trajectory_2.append(distance_2)
        except:
            trajectory_1.append(np.nan)
            trajectory_2.append(np.nan)
    return trajectory_1, trajectory_2





all_data = []
all_data.append(MilitaryList)
all_data.append(CivilianList)
fs = 100



def get_all_trajectories():
    data_dictionary = pd.DataFrame(columns = ["group", "pair", "order", "interactive", "sync", "hierarchy", "who_leader", "trial", "num_tries", "completion_time"
                                        "phase1", "phase2", "sync_crossings1", "desync_crossings1",  "sync_crossings2", "desync_crossings2", "startindex", "KOP_100", "KOP_20", "PLV_100", "PLV_20"])
    for Li in range(2):
        data = all_data[Li]
        if Li == 0:
            group = 'military'
        else:
            group = 'civilian'
        paircounter = 0
        num_tries = 0
        for pair in data:
            for trial in pair.TrialList:
                num_tries +=1
                if trial.Success == True:
                    
                    sync = False
                    if (trial.Condition == 'Sync_Egalitarian') or (trial.Condition == 'Sync_LF') or (trial.Condition == 'Sync_FL'):
                        sync = True    

                    hierarchy = True
                    if (trial.Condition == 'Sync_Egalitarian') or (trial.Condition == 'Desync_Egalitarian'):
                        hierarchy = False

                    if trial.Condition == 'Sync_Solo':
                        interactive = False
                        sync = True
                        hierarchy = False
                    else:
                        interactive = True

                    if "LF" in trial.Condition:
                        who_leader = 1
                    elif "FL" in trial.Condition:
                        who_leader = 2
                    else:
                        who_leader = 0
                    
                    CheckCount = 0
                    DesyncCount = 0

                    sync_crossings1 = dict()
                    desync_crossings1 = dict()
                    sync_crossings2 = dict()
                    desync_crossings2 = dict()


                    # check in each trial when there is a checkpoint being crossed
                    Checkpos_x_1, Checkpos_y_1, Desyncpos_x_1, Desyncpos_y_1, Checkpos_x_2, Checkpos_y_2, Desyncpos_x_2, Desyncpos_y_2 = load_points(
                        trial.Condition, trial.TrialNumber)
                    

                    startindex = 0
                    movement_start = False

                    grad_1_x = np.gradient(np.asarray(trial.Player_1_x))
                    grad_2_x = np.gradient(np.asarray(trial.Player_2_x))



                    grad_1_y = np.gradient(np.asarray(trial.Player_1_y))
                    grad_2_y = np.gradient(np.asarray(trial.Player_2_y))

                    speed_1 = Moving_average(np.sqrt(np.square(grad_1_x) + np.square(grad_1_y)),20)
                    speed_2 = Moving_average(np.sqrt(np.square(grad_2_x) + np.square(grad_2_y)),20)

                    analytic_signal1 = hilbert(speed_1)
                    analytic_signal2 = hilbert(speed_2)
                    full_phase1 = np.angle(analytic_signal1)
                    full_phase2 = np.angle(analytic_signal2)


                    # get the moment when movement is started
                    while movement_start == False:
                        startindex+=1
                        try:
                            if grad_1_x[startindex] != 0 or grad_2_x[startindex] != 0 or grad_1_y[startindex] != 0 or grad_1_y[startindex] != 0:
                                movement_start = True
                        except:
                            break
                    if movement_start == False:
                        continue

                    

                   
                    
                    # calculate PLV and KOP only from the moment the movement is started 
                    # but the phase should be for the complete signal so that it can be linked to the EEG data
                    speed_1 = speed_1[startindex:-1]
                    speed_2 = speed_2[startindex:-1]
                    truncated_time = trial.time[startindex:-1]

                    
                    

                    
                    _, _, PLV_20 = calculate_average_PLV(speed_1, speed_2, 20, 10)
                    _, _, KOP_20 = calculate_average_KOP(speed_1, speed_2, 20, 10)

                    _, _, PLV_100 = calculate_average_PLV(speed_1, speed_2, 100, 50)
                    _, _, KOP_100 = calculate_average_KOP(speed_1, speed_2, 100, 50)



                    # safeguard to avoid large movement artefacts in the beginning?
                    Threshold_passed = False
                    for index in range(0,len(grad_1_x)-1):
                        if np.abs(grad_1_x[index]) > 2 or np.abs(grad_2_x[index]) > 2 or np.abs(
                            grad_2_y[index]) > 2 or np.abs(grad_2_y[index]) > 2:
                            startindex = index + 10


                    # get the corssings of all the checkpoints
                    if sync:
                        for checkpoint in range(len(Checkpos_x_1)):
                            point_1 = [float(Checkpos_x_1[checkpoint]), float(Checkpos_y_1[checkpoint])]
                            point_2 = [float(Checkpos_x_2[checkpoint]), float(Checkpos_y_2[checkpoint])]
                            index_1 = 0
                            ok = True
                            crossed = False
                            while crossed == False:
                                index_1 += 1
                                try:
                                    trial.Player_1_x[index_1 + 1]
                                    trial.Player_2_x[index_1 + 1]
                                except:
                                    ok = False
                                    break
                                distance_1 = distance([trial.Player_1_x[index_1], trial.Player_1_y[index_1]], point_1)
                                distance_2 = distance([trial.Player_2_x[index_1], trial.Player_2_y[index_1]], point_2)

                                if ok == True:


                                    if distance_1 < 0.7:
                                        sync_crossings1[checkpoint+1] = index_1

                                    elif distance_2 < 0.7:
                                        sync_crossings2[checkpoint+1] = index_1
                                    

                    else: # desync
                        # do the same for the desynchrony points
                        for checkpoint in range(len(Desyncpos_x_1)):
                            point_1 = [float(Desyncpos_x_1[checkpoint]),
                                        float(Desyncpos_y_1[checkpoint])]
                            point_2 = [float(Desyncpos_x_2[checkpoint]),
                                        float(Desyncpos_y_2[checkpoint])]
                            index_1 = 0
                            ok = True
                            crossed = False
                            while crossed == False:
                                index_1 += 1
                                try:
                                    trial.Player_1_x[index_1 + 1]
                                    trial.Player_2_x[index_1 + 1]
                                except:
                                    ok = False
                                    break
                                distance_1 = distance([trial.Player_1_x[index_1], trial.Player_1_y[index_1]],
                                                        point_1)
                                distance_2 = distance([trial.Player_2_x[index_1], trial.Player_2_y[index_1]],
                                                        point_2)

                                if ok == True:
                                    if distance_1 < 0.7:
                                        desync_crossings1[checkpoint+1] = index_1

                                    elif distance_2 < 0.7:
                                       desync_crossings2[checkpoint+1] = index_1
                    

                    new_row = {"group": group, "pair": pair.Pair, "order": 0, "interactive": interactive, "sync": sync, "hierarchy": hierarchy, "who_leader": who_leader, "trial": trial.TrialNumber, "num_tries": num_tries,
                                            "completion_time": trial.CompletionTime, "phase1": full_phase1, "phase2": full_phase2, "sync_crossings1": sync_crossings1, "desync_crossings1": desync_crossings1,
                                                "sync_crossings2": sync_crossings2, "desync_crossings2": desync_crossings2, "startindex": startindex, 
                                                "KOP_100": KOP_100, "KOP_20": KOP_20, "PLV_100":PLV_100, "PLV_20": PLV_20}
                    data_dictionary = data_dictionary._append(new_row, ignore_index=True)
                    num_tries = 0



    with open(r"Behavioral_Dataframe.pickle", "wb") as output_file:
        pickle.dump(data_dictionary, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")


    return data_dictionary

# uncomment this to get trajectories
data_dictionary = get_all_trajectories()

with open(r"Behavioral_Dataframe.pickle", "rb") as input_file:
    data_dictionary = pickle.load(input_file)


# Filter out columns with unwanted data types and select specific columns
filtered_columns = ['group', 'pair', 'order', 'interactive', 'sync', 'hierarchy', "who_leader", 'trial', 'num_tries', 'completion_time', 'KOP_100', 'KOP_20', 'PLV_100', 'PLV_20']

# Write DataFrame to Excel file
with pd.ExcelWriter("data_dictionary.xlsx") as writer:
    # Ensure there's at least one row of data
    if not data_dictionary.empty:
        data_to_write = data_dictionary[filtered_columns]
        data_to_write.to_excel(writer, index=False, sheet_name="Sheet1")