import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert


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
    index = index - 20

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

            
                                            



def get_all_trajectories():
    distance_trajectories = pd.DataFrame(columns = ["KOP_trajectory", "PLV_trajectory", "pair", "success", "sync", "hierarchy", "who_first", "who_leader"])
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

                        PLV, interval_times, mean_PLV = calculate_average_PLV(speed_1[startindex:], speed_2[startindex:], 20, 1)
                        KOP, interval_times, mean_KOP = calculate_average_KOP(speed_1[startindex:], speed_2[startindex:], 20, 1)

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
                                            PLV_trajectory, KOP_trajectory = get_phase_measures(PLV, KOP, index_1)

                                            

                                            # insert here procedure to get the plv and kop

                                            new_row = {"KOP_trajectory": KOP_trajectory, "PLV_trajectory": PLV_trajectory,
                                                       "group": group, "pair": pair.Pair,
                                                       "success": True, "sync": True, "hierarchy": hierarchy,
                                                       "who_first": who_first,
                                                       "who_leader": who_leader}
                                            distance_trajectories = distance_trajectories._append(new_row,
                                                                                                 ignore_index=True)
                                        elif distance_2 < 0.7:
                                            who_leader = 'None'
                                            who_first = 'None'
                                            crossed = True
                                            if (condition == 'Sync_FL'):
                                                who_first = 'leader'
                                                who_leader = 2
                                            if (condition == 'Sync_LF'):
                                                who_first = 'follower'
                                                who_leader = 1

                                            # get time locked trajectories if player one is first
                                            PLV_trajectory, KOP_trajectory = get_phase_measures(PLV, KOP, index_1)

                                            # insert here procedure to get the plv and kop

                                            new_row = {"KOP_trajectory": KOP_trajectory, "PLV_trajectory": PLV_trajectory,
                                                       "success": True, "sync": True, "hierarchy": hierarchy, "who_first": who_first,
                                                       "who_leader": who_leader}
                                            distance_trajectories = distance_trajectories._append(new_row, ignore_index=True)

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
                                            PLV_trajectory, KOP_trajectory = get_phase_measures(PLV, KOP, index_1)

                                            # insert here procedure to get the plv and kop

                                            new_row = {"KOP_trajectory": KOP_trajectory, "PLV_trajectory": PLV_trajectory,
                                                       "group": group, "pair": pair.Pair,
                                                       "success": True, "sync": False, "hierarchy": hierarchy,
                                                       "who_first": who_first,
                                                       "who_leader": who_leader}
                                            distance_trajectories = distance_trajectories._append(new_row,
                                                                                                 ignore_index=True)

                                        elif distance_2 < 0.7:
                                            who_leader = 'None'
                                            who_first = 'None'
                                            crossed = True
                                            if (condition == 'Desync_FL'):
                                                who_first = 'leader'
                                                who_leader = 2
                                            if (condition == 'Desync_LF'):
                                                who_first = 'follower'
                                                who_leader = 1

                                            # get time locked trajectories if player one is first
                                            PLV_trajectory, KOP_trajectory = get_phase_measures(PLV, KOP, index_1)

                                            # insert here procedure to get the plv and kop

                                            new_row = {"KOP_trajectory": KOP_trajectory, "PLV_trajectory": PLV_trajectory,
                                                       "group": group, "pair": pair.Pair,
                                                       "success": True, "sync": False, "hierarchy": hierarchy, "who_first": who_first,
                                                       "who_leader": who_leader}
                                            distance_trajectories = distance_trajectories._append(new_row, ignore_index=True)





    with open(r"TimeLockedPhaseTrajectories.pickle", "wb") as output_file:
        pickle.dump(distance_trajectories, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")


    return distance_trajectories

# uncomment this to get trajectories
#distance_trajectories = get_all_trajectories()

with open(r"TimeLockedPhaseTrajectories.pickle", "rb") as input_file:
    distance_trajectories = pickle.load(input_file)




# make the 4 trajectory plots (for the 4 conditions)
def plot_condition_trajectories(ax, distance_trajectories, parameters):
    # first make the plot titles
    title = ''
    for key in parameters.keys():
        if parameters[key]:
            title = title + str(key)

    # first plot the successful trials
    parameters["group"] = 'military'
    print(parameters)
    select_trajectories = distance_trajectories
    for key in parameters :
        select_trajectories = select_trajectories[select_trajectories[key] == parameters[key]]
    print(select_trajectories)

    traj_PLV = select_trajectories.PLV_trajectory.to_list()
    first_trajectories =np.array([np.array(xi) for xi in traj_PLV])

    traj_KOP = select_trajectories.KOP_trajectory.to_list()
    second_trajectories =np.array([np.array(xi) for xi in traj_KOP])

    mean_first_trajectories = np.nanmean(first_trajectories, axis = 0)
    mean_second_trajectories = np.nanmean(second_trajectories, axis = 0)
    x = np.linspace(-1000,500,len(mean_second_trajectories))

    ax.plot(x, mean_first_trajectories, color = 'green', linestyle = '-', alpha = 0.8)
    ax.plot(x, mean_second_trajectories, color = 'green', linestyle = '--', alpha = 0.8)

    parameters["group"] = 'civilian'
    print(parameters)
    select_trajectories = distance_trajectories
    for key in parameters :
        select_trajectories = select_trajectories[select_trajectories[key] == parameters[key]]
    print(select_trajectories)

    traj_PLV = select_trajectories.PLV_trajectory.to_list()
    first_trajectories =np.array([np.array(xi) for xi in traj_PLV])

    traj_KOP = select_trajectories.KOP_trajectory.to_list()
    second_trajectories =np.array([np.array(xi) for xi in traj_KOP])

    mean_first_trajectories = np.nanmean(first_trajectories, axis = 0)
    mean_second_trajectories = np.nanmean(second_trajectories, axis = 0)
    x = np.linspace(-1000,500,len(mean_second_trajectories))
   
    ax.plot(x, mean_first_trajectories, color = 'orange', linestyle = '-', alpha = 0.8)
    ax.plot(x, mean_second_trajectories, color = 'orange', linestyle = '--', alpha = 0.8)


    ax.plot(x, 0.7*np.ones(len(mean_first_trajectories)), color = 'black')
    ax.axhline(y=0.7, color='black', linestyle='-')
    ax.axvline(x=0, color='black', linestyle='-')
    ax.axvline(x=200, color='black', linestyle='--', linewidth = 1)
    ax.set_title(title)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
parameters = {"sync": True, "hierarchy": False}
plot_condition_trajectories(ax1, distance_trajectories, parameters)
parameters = {"sync": True, "hierarchy": True}
plot_condition_trajectories(ax2, distance_trajectories, parameters)
parameters = {"sync": False, "hierarchy": False}
plot_condition_trajectories(ax3, distance_trajectories, parameters)
parameters = {"sync": False, "hierarchy": True}
plot_condition_trajectories(ax4, distance_trajectories, parameters)
plt.show()

