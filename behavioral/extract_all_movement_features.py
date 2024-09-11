#########################
# Perform timelock analysis with aggregate data of all participants
#########################


import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
import seaborn as sns



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

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def get_phase_measures(PLV, KOP, index):
    # get time locked trajectories if player two is first
    PLV_trajectory = []
    KOP_trajectory = []

    # compensate for the fact that plv and kop have one window length lag
    #index = index - 20

    for traj_index in range(index - 100, index + 50):
        try:
            PLV_trajectory.append(PLV[traj_index])
            KOP_trajectory.append(KOP[traj_index])
        except:
            PLV_trajectory.append(np.nan)
            KOP_trajectory.append(np.nan)
    return PLV_trajectory, KOP_trajectory

def get_middle_values(arr, num_values=100):
    # Ensure the array is at least as long as the desired number of values
    if len(arr) < num_values:
        raise ValueError("Array is too short to extract the desired number of middle values.")
    
    # Calculate the start and end indices
    start_index = (len(arr) - num_values) // 2
    end_index = start_index + num_values
    
    # Extract the middle values
    middle_values = arr[start_index:end_index]
    
    return middle_values

def get_cross_correlation(speed_1_x, speed_1_y, speed_2_x, speed_2_y):
    # get time locked trajectories if player two is first
    correlation_array = []

    try:
        correlation_array_x = np.correlate(speed_1_x, speed_2_x, mode = 'same')
        correlation_array_y = np.correlate(speed_1_y, speed_2_y, mode = 'same')

        correlation_array = (correlation_array_x + correlation_array_y) / 2

        correlation_array = get_middle_values(correlation_array , 200)

    except:
        'nothing'

    if len(correlation_array) != 200:
        correlation_array = np.zeros((200,))
        correlation_array[:] = np.nan

    
    return correlation_array



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


# Function to calculate the percentage position of each point on the individual trajectory
def calculate_percentage_along_avg_trajectory(x_coords, y_coords, avg_x_coords, avg_y_coords, total_length_avg, cumulative_distances_avg):
    percentages = []
    for x, y in zip(x_coords, y_coords):
        # Calculate distances from the point to each point on the average trajectory
        point_distances = np.sqrt((avg_x_coords - x)**2 + (avg_y_coords - y)**2)
        
        # Find the index of the closest point on the average trajectory
        closest_index = np.argmin(point_distances)
        
        # Calculate the cumulative distance up to the closest point
        distance_to_point = cumulative_distances_avg[closest_index - 1] if closest_index > 0 else 0
        
        # Calculate the percentage of the total length
        percentage_of_length = (distance_to_point / total_length_avg) * 100
        
        percentages.append(percentage_of_length)
    
    return np.array(percentages)

                    # 5. Transform percentage differences into time differences
def calculate_time_difference(percentages_traj_1, percentages_traj_2, percentage_diff, time,):
    time_diffs = []
    for i in range(len(percentage_diff)):
        # Find the corresponding time for each percentage
        # Assuming linear mapping between time and percentage:
        t1 = np.interp(percentages_traj_1[i], np.linspace(0, 100, len(time)), time)
        t2 = np.interp(percentages_traj_2[i], np.linspace(0, 100, len(time)), time)
        
        # Calculate the time difference
        time_diff = t1 - t2
        time_diffs.append(time_diff)
    
    return np.array(time_diffs)

def calculate_time_differences(trial):
    grad_1_x = np.gradient(np.asarray(trial.Player_1_x))
    grad_2_x = np.gradient(np.asarray(trial.Player_2_x))



    grad_1_y = np.gradient(np.asarray(trial.Player_1_y))

    
    startindex = 0
    movement_start = False

    while movement_start == False:
        startindex+=1
        try:
            if grad_1_x[startindex] != 0 or grad_2_x[startindex] != 0 or grad_1_y[startindex] != 0 or grad_1_y[startindex] != 0:
                movement_start = True
        except:
            break

    Threshold_passed = False
    for index in range(0,len(grad_1_x)-1):
        if np.abs(grad_1_x[index]) > 2 or np.abs(grad_2_x[index]) > 2 or np.abs(
            grad_2_x[index]) > 2 or np.abs(grad_2_x[index]) > 2:
            startindex = index + 10
    # Example trajectories (x and y coordinates)
    x_coords_1 = np.array(trial.Player_1_x[startindex:])
    y_coords_1 = np.array(trial.Player_1_y[startindex:])

    x_coords_2 = np.array(trial.Player_2_x[startindex:])
    y_coords_2 = np.array(trial.Player_2_y[startindex:]) + 5

    # 1. Calculate the average trajectory
    avg_x_coords = (x_coords_1 + x_coords_2) / 2
    avg_y_coords = (y_coords_1 + y_coords_2) / 2

    # 2. Calculate the cumulative distance along the average trajectory
    dx_avg = np.diff(avg_x_coords)
    dy_avg = np.diff(avg_y_coords)
    distances_avg = np.sqrt(dx_avg**2 + dy_avg**2)
    cumulative_distances_avg = np.cumsum(distances_avg)
    total_length_avg = cumulative_distances_avg[-1]

    # 3. Apply the function to each trajectory
    percentages_traj_1 = calculate_percentage_along_avg_trajectory(x_coords_1, y_coords_1, avg_x_coords, avg_y_coords, total_length_avg, cumulative_distances_avg)
    percentages_traj_2 = calculate_percentage_along_avg_trajectory(x_coords_2, y_coords_2, avg_x_coords, avg_y_coords, total_length_avg, cumulative_distances_avg)

    # 4. Calculate the difference in percentages for each point
    percentage_diff = percentages_traj_1 - percentages_traj_2

    time = np.array(trial.time[startindex:])


    # Calculate time differences
    time_diffs = calculate_time_difference(percentages_traj_1, percentages_traj_2, percentage_diff, time)

    return time_diffs, avg_x_coords, avg_y_coords
                                            


all_data = []
all_data.append(MilitaryList)
all_data.append(CivilianList)
fs = 100


"""
# test how much signal shift there is:
for Li in range(2):
    data = all_data[Li]
    if Li == 0:
        group = 'military'
    else:
        group = 'civilian'
    paircounter = 0
    for pair in data:
        for condition in ['Sync_Egalitarian', 'Sync_LF', 'Sync_FL','Desync_Egalitarian', 'Desync_LF', 'Desync_FL']:
            for trial in pair.TrialList:

                if trial.Success != True:
                    continue

                if trial.Condition == condition:
                    print(condition)
                    sync = False
                    if (condition == 'Sync_Egalitarian') or (condition == 'Sync_LF') or (condition == 'Sync_FL'):
                        sync = True

                    print('raw_len')
                    print(len(trial.Player_1_x))
                    grad_1_x = np.gradient(np.asarray(trial.Player_1_x))
                    grad_2_x = np.gradient(np.asarray(trial.Player_2_x))

                    print('grad_len')
                    print(len(grad_1_x))
                    grad_1_y = np.gradient(np.asarray(trial.Player_1_y))
                    grad_2_y = np.gradient(np.asarray(trial.Player_2_y))

                    speed_1 = Moving_average(np.sqrt(np.square(grad_1_x) + np.square(grad_1_y)),20)
                    speed_2 = Moving_average(np.sqrt(np.square(grad_2_x) + np.square(grad_2_y)),20)

                    print('speed_len')
                    print(len(speed_1))

                    
                    startindex = 0
                    movement_start = False


                    PLV, interval_times, mean_PLV = calculate_average_PLV(speed_1[startindex:], speed_2[startindex:], 20, 1)
                    KOP, interval_times, mean_KOP = calculate_average_KOP(speed_1[startindex:], speed_2[startindex:], 20, 1)

                    print('kop_len')
                    print(len(KOP))
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
"""
            
                                            



distance_trajectories = pd.DataFrame(columns = [ "pair", "condition", "success", "sync", "hierarchy", "who_first", "who_leader"])
for Li in range(2):
    data = all_data[Li]
    if Li == 0:
        group = 'military'
    else:
        group = 'civilian'
    paircounter = 0
    for pair in data:
        for condition in ['Individual', 'Sync_Egalitarian', 'Sync_LF', 'Sync_FL','Desync_Egalitarian', 'Desync_LF', 'Desync_FL']:
            for trial in pair.TrialList:

                if trial.Success != True:
                    continue

                if trial.Condition == condition:
                    #print(condition)
                    sync = False
                    if (condition == 'Sync_Egalitarian') or (condition == 'Sync_LF') or (condition == 'Sync_FL'):
                        sync = True

                    hierarchy = True
                    if (condition == 'Sync_Egalitarian') or (condition == 'Desync_Egalitarian'):
                        hierarchy = False
                    CheckCount = 0
                    DesyncCount = 0
                    # check in each trial when there is a checkpoint being crossed
                    Checkpos_x_1, Checkpos_y_1, Desyncpos_x_1, Desyncpos_y_1, Checkpos_x_2, Checkpos_y_2, Desyncpos_x_2, Desyncpos_y_2 = load_points(
                        condition, trial.TrialNumber)
                    

                    startindex = 0
                    movement_start = False

                    grad_1_x = np.gradient(np.asarray(trial.Player_1_x))
                    grad_2_x = np.gradient(np.asarray(trial.Player_2_x))



                    grad_1_y = np.gradient(np.asarray(trial.Player_1_y))
                    grad_2_y = np.gradient(np.asarray(trial.Player_2_y))

                    speed_1 = Moving_average(np.sqrt(np.square(grad_1_x) + np.square(grad_1_y)),20)
                    speed_2 = Moving_average(np.sqrt(np.square(grad_2_x) + np.square(grad_2_y)),20)

                    speed_1 = lowpass_filter(np.sqrt(np.square(grad_1_x) + np.square(grad_1_y)),10, 100, 5)
                    speed_2 = lowpass_filter(np.sqrt(np.square(grad_2_x) + np.square(grad_2_y)),10, 100, 5)

                    speed_1_x = lowpass_filter(grad_1_x,10, 100, 5)
                    speed_2_x = lowpass_filter(grad_2_x,10, 100, 5)

                    speed_1_y = lowpass_filter(grad_1_y,10, 100, 5)
                    speed_2_y = lowpass_filter(grad_2_y,10, 100, 5)
                        
                    #plt.show()
                    analytic_signal1 = hilbert(speed_1[startindex+20:-20])
                    analytic_signal2 = hilbert(speed_2[startindex+20:-20])

                    # Extract phase
                    phase1 = np.angle(analytic_signal1)
                    phase2 = np.angle(analytic_signal2)
                    phase_diff = phase2 - phase1

                    #axs[1].plot(trial.time[startindex+20:-20], phase_diff)
                    #plt.show()

                    phase_matrix = np.vstack((phase1, phase2))
                    KOP_raw = np.abs(np.mean(np.exp(1j * phase_matrix), 0))


                    PLV = phase_diff
                    KOP = np.abs(phase_diff)#KOP_raw
                    

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

                    if sync:
                        
                        who_leader = 'None'
                        who_first = 'None'
                        crossed = True
                        if (condition == 'Sync_LF'):
                            who_first = 'leader'
                            who_leader = 1
                        elif (condition == 'Sync_FL'):
                            who_first = 'follower'
                            who_leader = 2

                        # get time locked trajectories if player one is first
                        PLV_trajectory = 0
                        KOP_trajectory = 0

                        cross_correlation = get_cross_correlation(speed_2_x, speed_2_y, speed_1_x, speed_1_y)
                        if who_leader == 2:
                            cross_correlation = get_cross_correlation(speed_1_x, speed_1_y, speed_2_x, speed_2_y)                                               
                    
                        time_differences, _, _ = calculate_time_differences(trial)
                        # if player 1 is first than we should invert it so that the phase difference shows how much player one 'leads' before player 2
                        #PLV_trajectory = [-x for x in PLV_trajectory]

                        # insert here procedure to get the plv and kop

                        new_row = {"KOP_trajectory": KOP_trajectory, "PLV_trajectory": PLV_trajectory, "cross_correlation": cross_correlation, "time_differences": time_differences,
                                    "group": group, "pair": pair.Pair,
                                    "success": True, "sync": True, "hierarchy": hierarchy,
                                    "who_first": who_first,
                                    "who_leader": who_leader}
                        distance_trajectories = distance_trajectories._append(new_row,
                                                                                ignore_index=True)
                                    

                    else: # desync
                        # do the same for the desynchrony points
                        
                        who_leader = 'None'
                        who_first = 'None'
                        crossed = True
                        if (condition == 'Desync_LF'):
                            who_first = 'leader'
                            who_leader = 1
                        elif (condition == 'Desync_FL'):
                            who_first = 'follower'
                            who_leader = 2

                        # get time locked trajectories if player one is first

                        # get time locked trajectories if player one is first
                        PLV_trajectory = 0
                        KOP_trajectory = 0

                        cross_correlation = get_cross_correlation(speed_2_x, speed_2_y, speed_1_x, speed_1_y)
                        if who_leader == 2:
                            cross_correlation = get_cross_correlation(speed_1_x, speed_1_y, speed_2_x, speed_2_y)                                               

                        time_differences, _, _ = calculate_time_differences(trial)
                        
            
                        # insert here procedure to get the plv and kop

                        new_row = {"KOP_trajectory": KOP_trajectory, "PLV_trajectory": PLV_trajectory, "cross_correlation": cross_correlation, "time_differences": time_differences,
                                    "group": group, "pair": pair.Pair,
                                    "success": True, "sync": False, "hierarchy": hierarchy,
                                    "who_first": who_first,
                                    "who_leader": who_leader}
                        print('time diff')
                        #print(new_row["time_differences"])
                        distance_trajectories = distance_trajectories._append(new_row,
                                                                                ignore_index=True)
                        print(distance_trajectories.time_differences)
                    







