import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns


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




all_data = []
all_data.append(MilitaryList)
all_data.append(CivilianList)
fs = 100


def get_all_trajectories():
    distance_trajectories = pd.DataFrame(columns = ["first_trajectory", "second_trajectory", "pair", "success", "sync", "hierarchy", "who_first", "who_leader"])
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
                                            first_traj, second_traj = get_distance_trajectory(trial, index_1, point_1, point_2)
                                            new_row = {"first_trajectory": first_traj, "second_trajectory": second_traj,
                                                       "group": group, "pair": pair.Pair,
                                                       "success": True, "sync": True, "hierarchy": hierarchy,
                                                       "who_first": who_first,
                                                       "who_leader": who_leader}
                                            distance_trajectories = distance_trajectories.append(new_row,
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

                                            # get time locked trajectories if player two is first
                                            second_traj, first_traj = get_distance_trajectory(trial, index_1, point_1, point_2)
                                            new_row = {"first_trajectory": first_traj, "second_trajectory": second_traj, "group": group, "pair": pair.Pair,
                                                       "success": True, "sync": True, "hierarchy": hierarchy, "who_first": who_first,
                                                       "who_leader": who_leader}
                                            distance_trajectories = distance_trajectories.append(new_row, ignore_index=True)

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

                                            first_traj, second_traj = get_distance_trajectory(trial, index_1, point_1, point_2)
                                            new_row = {"first_trajectory": first_traj, "second_trajectory": second_traj,
                                                       "group": group, "pair": pair.Pair,
                                                       "success": True, "sync": False, "hierarchy": hierarchy,
                                                       "who_first": who_first,
                                                       "who_leader": who_leader}
                                            distance_trajectories = distance_trajectories.append(new_row,
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

                                                # get time locked trajectories if player two is first
                                            second_traj, first_traj = get_distance_trajectory(trial, index_1, point_1, point_2)
                                            new_row = {"first_trajectory": first_traj, "second_trajectory": second_traj,
                                                       "group": group, "pair": pair.Pair,
                                                       "success": True, "sync": False, "hierarchy": hierarchy, "who_first": who_first,
                                                       "who_leader": who_leader}
                                            distance_trajectories = distance_trajectories.append(new_row, ignore_index=True)

                        # if the trial is unsuccesful, the mistake is at the last checkpoint
                        # indicate this in the dataframe
                        if trial.Success == False:
                            # check if failure is due to point by seeing if the are near it at end of trial
                            if sync == True:
                                for checkpoint in range(len(Checkpos_x_1)):
                                    point_1 = [float(Checkpos_x_1[checkpoint]), float(Checkpos_y_1[checkpoint])]
                                    point_2 = [float(Checkpos_x_2[checkpoint]), float(Checkpos_y_2[checkpoint])]
                                    distance_1 = distance([trial.Player_1_x[-1], trial.Player_1_y[-1]],
                                                          point_1)
                                    distance_2 = distance([trial.Player_2_x[-1], trial.Player_2_y[-1]],
                                                          point_2)
                                    if (distance_1 < 1) or (distance_2 < 1):
                                        distance_trajectories.iloc[-1, distance_trajectories.columns.get_loc('success')] = False
                                        break #only want one point to be failure point
                            elif sync == False:
                                for checkpoint in range(len(Desyncpos_x_1)):
                                    point_1 = [float(Desyncpos_x_1[checkpoint]),
                                               float(Desyncpos_y_1[checkpoint])]
                                    point_2 = [float(Desyncpos_x_2[checkpoint]),
                                               float(Desyncpos_y_2[checkpoint])]
                                    distance_1 = distance([trial.Player_1_x[-1], trial.Player_1_y[-1]],
                                                          point_1)
                                    distance_2 = distance([trial.Player_2_x[-1], trial.Player_2_y[-1]],
                                                          point_2)
                                    # if one of the playesr is close to a desync point at the time of failure then take this as a failed desync point
                                    if (distance_1 < 1) or (distance_2 < 1):
                                        distance_trajectories.iloc[-1, distance_trajectories.columns.get_loc('success')] = False
                                        break #only want one point to be the failure point





    with open(r"TimeLockedTrajectories.pickle", "wb") as output_file:
        pickle.dump(distance_trajectories, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")


    return distance_trajectories

# uncomment this to get trajectories
#distance_trajectories = get_all_trajectories()

with open(r"TimeLockedTrajectories.pickle", "rb") as input_file:
    distance_trajectories = pickle.load(input_file)




# make the 4 trajectory plots (for the 4 conditions)
def plot_condition_trajectories(ax, distance_trajectories, parameters):
    # first make the plot titles
    title = ''
    for key in parameters.keys():
        if parameters[key]:
            title = title + str(key)

    # first plot the successful trials
    parameters["success"] = True
    print(parameters)
    select_trajectories = distance_trajectories
    for key in parameters :
        select_trajectories = select_trajectories[select_trajectories[key] == parameters[key]]
    print(select_trajectories)

    traj_1 = select_trajectories.first_trajectory.to_list()
    first_trajectories =np.array([np.array(xi) for xi in traj_1])

    traj_2 = select_trajectories.second_trajectory.to_list()
    second_trajectories =np.array([np.array(xi) for xi in traj_2])

    mean_first_trajectories = np.nanmean(first_trajectories, axis = 0)
    mean_second_trajectories = np.nanmean(second_trajectories, axis = 0)
    x = np.linspace(-1000,500,len(mean_second_trajectories))

    # and do the same for failed trials
    parameters["success"] = False
    select_trajectories = distance_trajectories
    for key in parameters :
        select_trajectories = select_trajectories[select_trajectories[key] == parameters[key]]
    traj_1 = select_trajectories.first_trajectory.to_list()
    first_trajectories =np.array([np.array(xi) for xi in traj_1])

    traj_2 = select_trajectories.second_trajectory.to_list()
    second_trajectories =np.array([np.array(xi) for xi in traj_2])

    mean_first_trajectories_failed = np.nanmean(first_trajectories, axis = 0)
    mean_second_trajectories_failed = np.nanmean(second_trajectories, axis = 0)

    ax.plot(x, mean_first_trajectories_failed, color = 'red', linestyle = '-', alpha = 0.8)
    ax.plot(x, mean_second_trajectories_failed, color = 'red', linestyle = '--', alpha = 0.8)

    ax.plot(x, mean_first_trajectories, color = 'green', linestyle = '-', alpha = 0.8)
    ax.plot(x, mean_second_trajectories, color = 'green', linestyle = '--', alpha = 0.8)

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


#now make the plot where you have leader - follower for every condition and do military and civilian graph


def plot_condition_difference_trajectory(ax, distance_trajectories, parameters):
    # first make the plot titles
    title = ''
    for key in parameters.keys():
        if parameters[key]:
            title = title + ' ' + str(key)

    # first plot the successful trials
    parameters["group"] = 'civilian'
    select_trajectories = distance_trajectories
    for key in parameters:
        select_trajectories = select_trajectories[select_trajectories[key] == parameters[key]]
    print(select_trajectories)

    traj_1 = select_trajectories.first_trajectory.to_list()
    first_trajectories = np.array([np.array(xi) for xi in traj_1])

    traj_2 = select_trajectories.second_trajectory.to_list()
    second_trajectories = np.array([np.array(xi) for xi in traj_2])

    difference_trajectories = first_trajectories - second_trajectories
    mean_difference_trajectories_civilian = np.nanmean(difference_trajectories, axis=0)

    x = np.linspace(-1000, 500, len(mean_difference_trajectories_civilian))

    # and do the same for failed trials
    parameters["group"] = 'military'
    select_trajectories = distance_trajectories
    for key in parameters:
        select_trajectories = select_trajectories[select_trajectories[key] == parameters[key]]
    traj_1 = select_trajectories.first_trajectory.to_list()
    first_trajectories = np.array([np.array(xi) for xi in traj_1])

    traj_2 = select_trajectories.second_trajectory.to_list()
    second_trajectories = np.array([np.array(xi) for xi in traj_2])

    difference_trajectories = first_trajectories - second_trajectories
    mean_difference_trajectories_military = np.nanmean(difference_trajectories, axis=0)

    ax.plot(x, mean_difference_trajectories_civilian, color='orange', linestyle='-', alpha=0.8)
    ax.plot(x, mean_difference_trajectories_military, color='green', linestyle='-', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-')
    ax.axvline(x=0, color='black', linestyle='-')
    ax.axvline(x=200, color='black', linestyle='--', linewidth=1)
    ax.set_title(title)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
parameters = {"sync": True, "hierarchy": False}
plot_condition_difference_trajectory(ax1, distance_trajectories, parameters)
parameters = {"sync": True, "hierarchy": True}
plot_condition_difference_trajectory(ax2, distance_trajectories, parameters)
parameters = {"sync": False, "hierarchy": False}
plot_condition_difference_trajectory(ax3, distance_trajectories, parameters)
parameters = {"sync": False, "hierarchy": True}
plot_condition_difference_trajectory(ax4, distance_trajectories, parameters)
plt.show()




def plot_going_first(ax, distance_trajectories, parameters):

    # first make the plot titles
    title = ''
    for key in parameters.keys():
        if parameters[key]:
            title = title + ' ' + str(key)

    # first plot the successful trials
    parameters["group"] = 'civilian'
    select_trajectories = distance_trajectories
    for key in parameters:
        select_trajectories = select_trajectories[select_trajectories[key] == parameters[key]]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(distance_trajectories.who_first)


    civilian_going_first = []
    for pair in select_trajectories.pair.unique():
        if pair not in [26, 32, 36, 39]:
            pair_trajectories = select_trajectories[select_trajectories['pair'] == pair]
            total_times = len(pair_trajectories)
            times_leader_first = len(pair_trajectories[pair_trajectories['who_first'] == 'leader'])
            print(times_leader_first)
            print(total_times)
            civilian_going_first.append(times_leader_first / total_times)

    # and do the same for failed trials
    parameters["group"] = 'military'
    select_trajectories = distance_trajectories
    for key in parameters:
        select_trajectories = select_trajectories[select_trajectories[key] == parameters[key]]

    military_going_first = []
    for pair in select_trajectories.pair.unique():
        pair_trajectories = select_trajectories[select_trajectories['pair'] == pair]
        total_times = len(pair_trajectories)
        times_leader_first = len(pair_trajectories[pair_trajectories['who_first'] == 'leader'])
        print(times_leader_first)
        print(total_times)
        military_going_first.append(times_leader_first / total_times)

    df = pd.melt(pd.DataFrame(
        {"civilian": np.array(civilian_going_first), "military": np.array(military_going_first)}), var_name='group', value_name='times leader first')
    sns.violinplot(data=df, x='group', y='times leader first', hue='group')
    plt.show()

parameters = {"sync": True, "hierarchy": True}
plot_going_first(ax1, distance_trajectories, parameters)

parameters = {"sync": False, "hierarchy": True}
plot_going_first(ax1, distance_trajectories, parameters)