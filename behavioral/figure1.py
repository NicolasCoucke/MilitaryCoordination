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



"""
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
"""



def load_points(condition, level):
    if int(level) > 10:
        _Level = 'Level' + str(int(level)-10)
    else:
        _Level = 'Level' + str(level)
    obstacle_path = os.path.join(path, 'LEVELDATA', condition, _Level)
    
    def load_csv(file_name):
        df = pd.read_csv(os.path.abspath(os.path.join(obstacle_path, file_name)), sep='delimiter', header=None, engine='python')
        pos_x = [string[1:].split(",")[0] for string in df[0]]
        pos_y = [string[1:].split(",")[1] for string in df[0]]
        return pos_x, pos_y
    
    Checkpos_x_1, Checkpos_y_1 = load_csv('CheckPointList_1.csv')
    Desyncpos_x_1, Desyncpos_y_1 = load_csv('DesyncPointList_1.csv')
    Checkpos_x_2, Checkpos_y_2 = load_csv('CheckPointList_2.csv')
    Desyncpos_x_2, Desyncpos_y_2 = load_csv('DesyncPointList_2.csv')
    
    # Load points from Mines.csv
    mines_file_path = os.path.join(obstacle_path, 'Mines.csv')
    if os.path.exists(mines_file_path):
        Mines_x, Mines_y = load_csv('Mines.csv')
    else:
        Mines_x, Mines_y = [], []

    if int(level) > 10:
        Checkpos_x_1 = [-float(x) for x in Checkpos_x_1]
        Checkpos_x_2 = [-float(x) for x in Checkpos_x_2]
        Desyncpos_x_1 = [-float(x) for x in Desyncpos_x_1]
        Desyncpos_x_2 = [-float(x) for x in Desyncpos_x_2]
        Mines_x = [-float(x) for x in Mines_x]

    return Checkpos_x_1, Checkpos_y_1, Desyncpos_x_1, Desyncpos_y_1, Checkpos_x_2, Checkpos_y_2, Desyncpos_x_2, Desyncpos_y_2, Mines_x, Mines_y



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


def get_cross_correlation(speed_1_x, speed_1_y, speed_2_x, speed_2_y, index):
    # get time locked trajectories if player two is first
    correlation_array = []

    try:
        correlation_array_x = np.correlate(speed_1_x[index-50: index+50], speed_2_x[index-50: index+50], mode = 'same')
        correlation_array_y = np.correlate(speed_1_y[index-50: index+50], speed_2_y[index-50: index+50], mode = 'same')

        correlation_array = (correlation_array_x + correlation_array_y) / 2
    except:
        'nothing'

    if len(correlation_array) != 100:
        correlation_array = np.zeros((100,))
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
    
    # Example trajectories (x and y coordinates)
    x_coords_1 = np.array(trial.Player_1_x[startindex:])
    y_coords_1 = np.array(trial.Player_1_y[startindex:])

    x_coords_2 = np.array(trial.Player_2_x[startindex:])
    y_coords_2 = np.array(trial.Player_2_y[startindex:]) + 5 # negative to put them on the same place

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

    return time_diffs, avg_x_coords, avg_y_coords, percentages_traj_1, percentages_traj_2
                                            



all_data = []
all_data.append(MilitaryList)
all_data.append(CivilianList)
fs = 100
                

folder_path = r"C:\Users\nicoucke\OneDrive - UGent\Desktop\Hyperscanning 1\paper plots"


distance_trajectories = pd.DataFrame(columns = ["KOP_trajectory", "PLV_trajectory", "cross_correlation", "pair", "success", "sync", "hierarchy", "who_first", "who_leader"])
for Li in range(2):
    data = all_data[Li]
    if Li == 0:
        group = 'military'
    else:
        group = 'civilian'
    paircounter = 0
    for pair in data:

        if pair.Pair != 1:
            continue


                # for plotting separate trajectories
        
        trialnumbers = [42, 183, 45, 184]

        fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))
        axs2 = axs2.flatten()
        #fig2.suptitle('Pair ' + str(pair.Pair))

        
        for condition in ['Sync_Egalitarian', 'Desync_LF']:# ['Sync_Egalitarian', 'Desync_Egalitarian']:

            
            


            current_trial = 1
            attempts = 0

            i = 0
            for trial in pair.TrialList:
                
                if current_trial == int(trial.TrialNumber):
                    attempts+=1

                i+=1
                print(i)

                if i in trialnumbers:
                    ax_i = trialnumbers.index(i)
                else:
                    continue
                
                #if trial.Success != True:
                #    continue
                
                

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
                    Checkpos_x_1, Checkpos_y_1, Desyncpos_x_1, Desyncpos_y_1, Checkpos_x_2, Checkpos_y_2, Desyncpos_x_2, Desyncpos_y_2, Mines_x, Mines_y = load_points(
                        condition, trial.TrialNumber)
                    

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

                    
                    startindex = 0
                    movement_start = False

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
                    

                    if 'De' in condition:
                        _color = 'orange'
                    else:
                        _color = 'green'


                   

                    """
                    fig, ax = plt.subplots()
                    values = np.array(time_diffs)
                    ax.set_xlim(-1,1)
                    sns.kdeplot(data=values, ax = ax)
                    plt.show()
                    
                    
                    
                    # Create a figure and an axis
                    fig, ax = plt.subplots()
                    """

                    

                    # Plot the circles
                    for x1, y1, x2, y2 in zip(Checkpos_x_1, Checkpos_y_1, Checkpos_x_2, Checkpos_y_2):
                        circle1 = plt.Circle((float(x1), float(y1)), radius=0.5, color='green', alpha = 0.4)
                        circle2 = plt.Circle((float(x2), float(y2)), radius=0.5, color='green', alpha = 0.4)
                        
                        # Add circles to the plot
                        axs2[ax_i].add_patch(circle1)
                        axs2[ax_i].add_patch(circle2)

                    for x1, y1, x2, y2 in zip(Desyncpos_x_1, Desyncpos_y_1, Desyncpos_x_2, Desyncpos_y_2):
                        
                        

                        circle1 = plt.Circle((float(x1), float(y1)), radius=0.5, color=_color, alpha = 0.4)
                        circle2 = plt.Circle((float(x2), float(y2)), radius=0.5, color=_color, alpha = 0.4)
                        
                        # Add circles to the plot
                        axs2[ax_i].add_patch(circle1)
                        axs2[ax_i].add_patch(circle2)

                    for x1, y1 in zip(Mines_x, Mines_y):
                        circle1 = plt.Circle((float(x1), float(y1)), radius=0.5, color='black', alpha = 0.4)                        
                        # Add circles to the plot
                        axs2[ax_i].add_patch(circle1)

                    axs2[ax_i].axhline(y=0, color = 'black', alpha = 0.5)
                    axs2[ax_i].set_xlim(-8, 7.5)
                    axs2[ax_i].set_ylim(-5.5,5.5)

                    axs2[ax_i].set_xticks([])
                    axs2[ax_i].set_yticks([])
                    #plt.plot(trial.Player_1_x[startindex:], trial.Player_1_y[startindex:], linewidth = 2)  
                    #plt.plot(trial.Player_1_x[startindex:], trial.Player_2_y[startindex:], linewidth = 2)  

                    axs2[ax_i].scatter(trial.Player_1_x[startindex:], trial.Player_1_y[startindex:], alpha = 0.05, s=70, edgecolors = 'none')  
                    axs2[ax_i].scatter(trial.Player_2_x[startindex:], trial.Player_2_y[startindex:], alpha = 0.05, s=70, edgecolors = 'none') 
                    
                    axs2[ax_i].set_aspect('equal', 'box')

                    #axs2[ax_i].set_title('Trial ' + str(trial.TrialNumber) + '; Attempts: ' + str(attempts))

                    attempts = 0
                    current_trial+=1

        fig2.tight_layout()
        fig2.savefig(os.path.join(folder_path, "figure_1_trajectories.pdf"))
        #plt.show()

                    # Show the first figure
                    #fig1.savefig(os.path.join(pair_folder_path, condition + "_merged_trajectories.png"))
                    #fig2.savefig(os.path.join(pair_folder_path, condition + "_both_trajectories.png"))


                  