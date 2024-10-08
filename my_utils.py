
import numpy as np
import mne
from autoreject import get_rejection_threshold, AutoReject
from mne.preprocessing import ICA, corrmap
#from mne_icalabel import label_components
import copy
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import mne
from autoreject import get_rejection_threshold, AutoReject
from mne.preprocessing import ICA, corrmap
import mne_icalabel
from mne_icalabel import label_components
import scipy
import scipy.signal as signal
import scipy.stats
import pandas as pd
#from astropy.stats import circmean



def extract_trials(events):
    trialial_info = []
    event_values = events[:, 2]
    event_samples = events[:,0]
    print(event_values)

    # find indices of trial start within the event matrix
    trial_start_indices = np.where(event_values < 10)[0]
    trial_counter = 1
    trials = np.empty(5,)
    for trial_start in trial_start_indices:
        j = trial_start
        end_trial = 0
        condition = event_values[trial_start]
        # don't care about tutorial trials
        if condition == 1:
            continue

        # j remains at start of trial while k goes to the end
        k = j

        while end_trial == 0:
            if event_values[k] == 150: # if player one finishes
                # check if the trial is a real succes
                jk = k
                secondfinished = False
                gameover = False

                # as long as the next trial doesnt start
                while event_values[jk] > 10:
                    # does the second player also finish?
                    if event_values[jk] == 250:
                        secondfinished = True
                    elif ( np.remainder(event_values[jk], 140 + condition) == 0 ) or ( np.remainder(event_values[jk], 240 + condition) == 0 ):
                        gameover = True
                    jk += 1
                    if jk > len(event_samples)-1:
                        break

                if (secondfinished == True) and (gameover == False):
                    success = True 
                else:
                    success = False

                end_trial = 1

                # if this was a real trial than add it, otherwise remove
            elif event_values[k] == 250:
                jk = k
                secondfinished = False
                gameover = False
                while event_values[jk] > 10:
                    # does the second player also finish?
                    if event_values[jk] == 150:
                        secondfinished = True
                    elif ( np.remainder(event_values[jk], 140 + condition) == 0 ) or ( np.remainder(event_values[jk], 240 + condition) == 0 ):
                        gameover = True
                    jk += 1
                    if jk > len(event_samples)-1:
                        break

                if (secondfinished == True) and (gameover == False):
                    success = True
                else:
                    success = False
                end_trial = 1
            # if one of them gets game over then they are both gameove
            elif ( np.remainder(event_values[k], 140 + condition) == 0 ) or ( np.remainder(event_values[k], 240 + condition) == 0 ):
                success = False
                end_trial = 1

            elif trial_start == trial_start_indices[-1]:
                end_trial = 1
            else:
                # if no gameover or finish then continue searching
                end_trial = 0
            
            k += 1
            if k > len(event_samples)-1:
                break
        
        if k > len(event_samples)-1:
            break

        trial_begin = event_samples[j]
        trial_end = event_samples[k]

        new_trial = [trial_begin, trial_end, trial_counter, condition, success]
        trials = np.vstack((trials, np.array(new_trial)))
        trial_counter +=1
    return trials



def create_sub_epochs(trials, sfreq, window_size, step_size):
    """
    # create epochs with sliding windows with length 1 and sliding 0.5
    """

    events = np.empty(3,)
    event_trialnumbers = np.empty(3,)
    for trial_index in range(np.size(trials,0)):
        trial_start = trials[trial_index, 0]
        trial_end = trials[trial_index, 1]
        condition = trials[trial_index, 3]
        trialnumber= trials[trial_index, 4]



        epoch_start = trial_start 
        while (epoch_start + window_size * sfreq ) < trial_end:
            event = [epoch_start, 0, condition]
            events = np.vstack((events, np.array(event)))
            
            event_trialnumbers = [epoch_start, 0, trialnumber]
            event_trialnumbers = np.vstack((event_trialnumbers, event_trialnumbers))

            # sliding window with 0.5s interval
            epoch_start += step_size * sfreq
    events = (np.rint(events)).astype(int)
    event_trialnumbers = (np.rint(event_trialnumbers)).astype(int)

    return events, event_trialnumbers


def create_time_locked_epochs(trials, sfreq, old_events):
    event_values = old_events[:, 2]
    event_samples = old_events[:,0]
    events = np.zeros((3,))
    event_info = np.zeros((3,)) # (trial number, successful, baseline)
    for trial_index in range(np.size(trials,0)):
        trial_start = trials[trial_index, 0]
        trial_end = trials[trial_index, 1]
        condition = trials[trial_index, 3]
        success = trials[trial_index, 4]

        """
        # first add the baseline (start of trial)
        event = [trial_start, 0, condition]
        events = np.vstack((events, np.array(event)))
        info = [trial_index, 0, 1]
        event_info = np.vstack((event_info, info))
        """

        # find the events happening within this trial
        trial_events = np.where((trial_start < event_samples) & (trial_end > event_samples))[0]
        
        
        previous_event = 0

        if condition in [2, 3, 4, 5]:
            # find the first cross of the checkpoint
            crossed_points = []
            
            for trial_event in trial_events:
                event_value = str(event_values[trial_event])
                    
                if len(event_value) < 3:
                    continue
                    
                player_crossed = int(event_value[0:2])
                checkpoint = int(event_value[2])


                if ((player_crossed == 21) or (player_crossed == 11)) and (checkpoint not in crossed_points):
                    # add checkpoint to list

                    event_sample = event_samples[trial_event]
                    # only add the event when the last one was not less than 500ms before so as to not mix them too much
                    if (event_sample - previous_event) > 0.5*sfreq:
                        event = [event_sample, 0, condition]
                        events = np.vstack((events, np.array(event)))
                        info = [trial_index, 1, 0]
                        event_info = np.vstack((event_info, info))
                        crossed_points.append(checkpoint)
                        previous_event = event_sample

                    # or if they are game over (i.e., failure)
                elif ((player_crossed == 24) or (player_crossed == 14)):
                    event_sample = event_samples[trial_event]
                    event = [event_sample, 0, condition]
                    events = np.vstack((events, np.array(event)))
                    info = [trial_index, 0, 0]
                    event_info = np.vstack((event_info, info))

                


        elif condition in [6,7,8]:
            # find the first cross of the desync
            crossed_points = []
            for trial_event in trial_events:
                event_value = str(event_values[trial_event])
                
                if len(event_value) < 3:
                    continue
                    
                player_crossed = int(event_value[0:2])
                checkpoint = int(event_value[2])

                # only for desync points
                if ((player_crossed == 20) or (player_crossed == 10)) and (checkpoint not in crossed_points):
                    # add checkpoint to list
                    if (event_sample - previous_event) > 0.5*sfreq:
                        event_sample = event_samples[trial_event]
                        event = [event_sample, 0, condition]
                        events = np.vstack((events, np.array(event)))

                        info = [trial_index, 1, 0]
                        event_info = np.vstack((event_info, info))
                        crossed_points.append(checkpoint)
                        previous_event = event_sample

                # or if they are game over
                elif ((player_crossed == 24) or (player_crossed == 14)):
                    event_sample = event_samples[trial_event]
                    event = [event_sample, 0, condition]
                    events = np.vstack((events, np.array(event)))

                    info = [trial_index, 0, 0]
                    event_info = np.vstack((event_info, info))


       
    events = (np.rint(events)).astype(int)
    event_info = (np.rint(event_info)).astype(int)

    events = events[1:,:]
    event_info = event_info[1:,:]

    return events, event_info  

def link_eeg_to_behavioral_trials(trials, data_dictionary, pair, sfreq):
    """
    only looks at successful trials
    for each trial in the behavioral data, looks if there is a corresponding eeg trial and labels it with the corresponding trialnumber
    eeg trials that did not find a match are labeled as nan
    the 'labels' are stored in the fourth column of the updated_trials matrix
    together with the condition number, these labels can then be used to link behavioral and eeg

    the trials matrix has the following 4 columns
    
    begin_sample, end_sample, eeg_trigger_index, eeg_trigger_value, trial_success

    """


    successes = np.array(trials[:,4], dtype = int)
    segments = trials[successes == 1,:]
    #segments = segments[:,:2]
    #new_trial = [trial_begin, trial_end, trial_counter, condition, success]
    updated_trials = segments
    updated_trials[:,4] = np.nan
    #updated_trial = [trial_begin, trial_end, trial_counter, condition, trial_number]

    event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 'Synchronous/FollowerLeader': 4, 'Individual': 5, 'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 'Complementary/FollowerLeader': 8}

    # get all data of the pair and then loop through the conditions
    for key, value in event_id.items():
        print(key)
       
        condition_sements = segments[segments[:,3] == value]
        

        if np.size(condition_sements, 0) == 0:
            break

        if value in [2, 3, 4, 5]:
            sync = True
        else:
            sync = False
        
        if value in [3, 4, 7, 8]:
            hierarchy = True    
        else:
            hierarchy = False

        if value == 5:
            interactive = False
        else:
            interactive = True

        if value in [3, 7]:
            who_leader = 1
        elif value in [4, 8]:
            who_leader = 2
        else:
            who_leader = 0

            # Filter rows based on conditions
        filtered_data = data_dictionary[(data_dictionary['pair'] == pair) &
                                        (data_dictionary['sync'] == sync) & 
                                        (data_dictionary['hierarchy'] == hierarchy) & 
                                        (data_dictionary['who_leader'] == who_leader) & 
                                        (data_dictionary['interactive'] == interactive)]
        
        eeg_index = 0
        fail_counter = 0
        eeg_indices_for_trials = np.zeros((20,))

        summary_for_condition = np.zeros((20, 6))
        summary_for_condition[:] = np.nan

        for trial in range(1, 21):

            summary_for_condition[trial-1,0] = trial

            if np.size(condition_sements, 0) == eeg_index:
                break  

            # Iterate through each trial number of the behavioral data
            for index, row in filtered_data.iterrows():

                # Check if the trial number matches the current trial in the loop
                if int(row['trial']) == trial:
                    # If a matching trial is found, print its completion time
                    completion_time = np.max([len(row['phase2'])/100, len(row['phase1'])/100])
                    summary_for_condition[trial-1,1] = completion_time

                    # then get the trial duration of what should be the corresponding eeg trial
                    eeg_time = (condition_sements[eeg_index,1] - condition_sements[eeg_index,0])/sfreq
                    summary_for_condition[trial-1,2] = eeg_index
                    summary_for_condition[trial-1,3] = eeg_time
                    # why this? Because it is training? or solo?
                    if value == 5:
                        eeg_indices_for_trials[trial-1] = eeg_index
                        #print(f" trial {trial} | behavioral: {completion_time} | EEG: {eeg_time}")
                        continue
                    
                    # if the durations match and also the index is not yet used, then add it to the indices 
                    if np.abs(eeg_time - completion_time) < 0.1:
                        
                        if eeg_index not in eeg_indices_for_trials:
                            fail_counter = 0
                            eeg_indices_for_trials[trial-1] = eeg_index
                            #print(f" trial {trial} | behavioral: {completion_time} | EEG: {eeg_time}")
                        else:
                            eeg_indices_for_trials[trial-1] = np.nan
                            #print(f" trial {trial} | behavioral: {completion_time} | EEG: {0}")
                    else:
                        # case when there is one eeg too many
                        # if the duration doesnt match for example 
                        next_eeg_index = eeg_index+1
                        if np.size(condition_sements, 0) == next_eeg_index:
                            eeg_indices_for_trials[trial-1] = np.nan
                            #print(f" trial {trial} | behavioral: {completion_time} | EEG: {0}")
                            break
                        next_eeg_time = (condition_sements[next_eeg_index,1] - condition_sements[next_eeg_index,0])/sfreq
                        if np.abs(next_eeg_time - completion_time) < 0.1:
                            eeg_indices_for_trials[trial-1] = next_eeg_index
                            summary_for_condition[trial-1,2] = next_eeg_index
                            summary_for_condition[trial-1,3] = next_eeg_time
                            #print(f" trial {trial} | behavioral: {completion_time} | EEG: {next_eeg_time}")
                        else:
                            # case where there is no eeg for the behavioral data
                            # go to next trial but keep current eeg index
                            eeg_indices_for_trials[trial-1] = np.nan
                            #print(f" trial {trial} | behavioral: {completion_time} | EEG: {0}")
                            fail_counter+= 1
                            if fail_counter < 2:
                                eeg_index-=1   



                    
            eeg_index+=1
        #print(eeg_indices_for_trials)
        # now we have the indices and we use them to find the trialcounter values that we wanted
        # and then next to the trialcounter values in the original trials object we put the trialnumber values

        print(summary_for_condition)

        for i in range(1, 21):
            eeg_index = eeg_indices_for_trials[i-1]
            if not np.isnan(eeg_index):
                
                # get the trial (eeg trigger) that corresponds to the trial
                trialcounter = condition_sements[int(eeg_index),2]
                #print(trialcounter)

                # in the updated trials, add the trialnumber on the 4th position
                updated_trials[np.where(updated_trials[:,2] == trialcounter)[0],4] = i

    return updated_trials



def link_eeg_to_behavioral_trials_via_excel(trials, data_dictionary, pair, sfreq, excel_writer):
    """
    Updates the function to save the summary_for_condition matrix for each condition in an Excel file.
    The condition name is placed above each matrix, and pair information is included for each entry.
    """
    
    successes = np.array(trials[:,4], dtype=int)
    segments = trials[successes == 1, :]
    updated_trials = segments
    updated_trials[:, 4] = np.nan

    event_id = {'Synchronous/Egalitarian': 2, 'Synchronous/LeaderFollower': 3, 
                'Synchronous/FollowerLeader': 4, 'Individual': 5, 
                'Complementary/Egalitarian': 6, 'Complementary/LeaderFollower': 7, 
                'Complementary/FollowerLeader': 8}
    
    if 'Sheet1' not in excel_writer.book.sheetnames:
        excel_writer.book.create_sheet('Sheet1')

    start_row = excel_writer.sheets['Sheet1'].max_row + 1 if excel_writer.sheets['Sheet1'].max_row > 0 else 0

     
    
    for key, value in event_id.items():
        print(key)
        
        condition_segments = segments[segments[:, 3] == value]
        
        if np.size(condition_segments, 0) == 0:
            continue
        
        sync = value in [2, 3, 4, 5]
        hierarchy = value in [3, 4, 7, 8]
        interactive = value != 5
        who_leader = 1 if value in [3, 7] else 2 if value in [4, 8] else 0

        filtered_data = data_dictionary[(data_dictionary['pair'] == pair) &
                                        (data_dictionary['sync'] == sync) & 
                                        (data_dictionary['hierarchy'] == hierarchy) & 
                                        (data_dictionary['who_leader'] == who_leader) & 
                                        (data_dictionary['interactive'] == interactive)]
        
        eeg_index = 0
        summary_for_condition = np.zeros((20, 6))
        summary_for_condition[:] = np.nan

        for trial in range(1, 21):
            summary_for_condition[trial-1, 0] = trial

            if np.size(condition_segments, 0) <= eeg_index:
                continue

            for index, row in filtered_data.iterrows():
                if int(row['trial']) == trial:
                    completion_time = np.max([len(row['phase2'])/100, len(row['phase1'])/100])
                    summary_for_condition[trial-1, 1] = completion_time
                    eeg_time = (condition_segments[eeg_index, 1] - condition_segments[eeg_index, 0]) / sfreq
                    summary_for_condition[trial-1, 2] = eeg_index
                    summary_for_condition[trial-1, 3] = eeg_time

            eeg_index += 1

        # Convert the matrix and associated metadata to a DataFrame
        df_condition = pd.DataFrame(summary_for_condition, columns=["Trial", "CompletionTime", "EEGIndex", "EEGTime", "Col5", "Col6"])
        df_condition.insert(0, 'Condition', key)
        df_condition.insert(0, 'Pair', pair)
        
        # Write the condition name, pair, and data to the Excel sheet
        df_condition.to_excel(excel_writer, sheet_name='Sheet1', startrow=start_row, index=False)
        start_row += len(df_condition) + 2  # Adjust row position for next condition (including space for a blank row)
    
    return updated_trials




def link_eeg_to_behavioral_trials_read_from_excel(trials, excel_path, pair, sfreq):
    """
    Function to link EEG trials to behavioral trials using data from a preprocessed Excel file.
    
    Parameters:
    - trials: numpy array containing EEG trial data with columns: [begin_sample, end_sample, eeg_trigger_index, eeg_trigger_value, trial_success]
    - excel_path: Path to the Excel file containing the preprocessed behavioral data.
    - pair: Identifier for the current pair being processed.
    - sfreq: Sampling frequency of the EEG data.
    
    Returns:
    - updated_trials: numpy array similar to 'trials' with an additional column linking EEG data to behavioral trial numbers.
    """

    # Read the Excel file to get the behavioral data for the specific pair
    df = pd.read_excel(excel_path, sheet_name='Sheet1')
    
    # Filter the data for the specific pair
    pair_data = df[df['Pair'] == pair]

    successes = np.array(trials[:, 4], dtype=int)
    segments = trials[successes == 1, :]
    updated_trials = segments.copy()
    updated_trials[:, 4] = np.nan

    # Iterate through the conditions as specified in the Excel data
    conditions = pair_data['Condition'].unique()

    for condition in conditions:
        condition_data = pair_data[pair_data['Condition'] == condition]
        
        event_id = condition_data.iloc[0]['EEGIndex']
        condition_segments = segments[segments[:, 3] == event_id]

        if condition_segments.size == 0:
            continue

        # Loop through each row in the condition data
        for idx, row in condition_data.iterrows():
            trial_number = int(row['Trial'])
            completion_time = row['CompletionTime']
            eeg_index = int(row['EEGIndex'])

            if np.isnan(eeg_index) or eeg_index >= len(condition_segments):
                continue

            # Calculate the EEG time
            eeg_time = (condition_segments[eeg_index, 1] - condition_segments[eeg_index, 0]) / sfreq

            # Check if the EEG time matches the completion time within a threshold
            if np.abs(eeg_time - completion_time) < 0.1:
                trialcounter = condition_segments[eeg_index, 2]
                updated_trials[np.where(updated_trials[:, 2] == trialcounter)[0], 4] = trial_number

    return updated_trials



def add_samples_to_behavioral_trials_from_excel(trials, excel_path, pair, sfreq):
    """
    Function to link EEG trials to behavioral trials using data from a preprocessed Excel file.
    
    Parameters:
    - trials: numpy array containing EEG trial data with columns: [begin_sample, end_sample, eeg_trigger_index, eeg_trigger_value, trial_success]
    - excel_path: Path to the Excel file containing the preprocessed behavioral data.
    - pair: Identifier for the current pair being processed.
    - sfreq: Sampling frequency of the EEG data.
    
    Returns:
    - updated_trials: numpy array similar to 'trials' with an additional column linking EEG data to behavioral trial numbers.
    """

    # Read the Excel file to get the behavioral data for the specific pair
    df = pd.read_excel(excel_path, sheet_name='Sheet1')
    
    # Filter the data for the specific pair
    pair_data = df[df['Pair'] == pair]

    successes = np.array(trials[:, 4], dtype=int)
    segments = trials[successes == 1, :]
    updated_trials = segments.copy()
    updated_trials[:, 4] = np.nan

    # Iterate through the conditions as specified in the Excel data
    conditions = pair_data['Condition'].unique()

    for condition in conditions:
        condition_data = pair_data[pair_data['Condition'] == condition]
        
        event_id = condition_data.iloc[0]['EEGIndex']
        condition_segments = segments[segments[:, 3] == event_id]

        if condition_segments.size == 0:
            continue

        # Loop through each row in the condition data
        for idx, row in condition_data.iterrows():
            trial_number = int(row['Trial'])
            completion_time = row['CompletionTime']
            eeg_index = int(row['EEGIndex'])

            if np.isnan(eeg_index) or eeg_index >= len(condition_segments):
                continue

            # Calculate the EEG time
            eeg_time = (condition_segments[eeg_index, 1] - condition_segments[eeg_index, 0]) / sfreq

            # Check if the EEG time matches the completion time within a threshold
            if np.abs(eeg_time - completion_time) < 0.1:
                trialcounter = condition_segments[eeg_index, 2]
                updated_trials[np.where(updated_trials[:, 2] == trialcounter)[0], 4] = trial_number

    return updated_trials



def get_channels_to_reject(spliced_raw, events):
     # set average reference
    spliced_raw_reref = spliced_raw.copy()
    spliced_raw_reref.set_eeg_reference(ref_channels='average', verbose = False)
      
    channel_powers = []
    psds, freqs = spliced_raw_reref.compute_psd(fmin = 1., fmax = 50., n_fft = 2048).get_data(return_freqs = True)
    
    for channel_index in range(64):
        psd_channel = 10 * np.log10(psds[channel_index,:]*10**12)
        average_power = np.mean(psd_channel)
        channel_powers.append(average_power) # microvolt squared


    
    # detect channel powers that are too large
    channel_powers = np.array(channel_powers)
    rejected_channels_power = np.where((np.abs(channel_powers - np.mean(channel_powers)) > 3*np.std(channel_powers))) #np.where(channel_powers > 200) #np.where(np.abs(channel_powers - np.mean(channel_powers)) > 2*np.std(channel_powers))

    rejected_channels = rejected_channels_power[0]


    # mark those channels as bad in the raw object
    
   
    del spliced_raw_reref 
   
    return rejected_channels

def ICA_autocorrect(icas: list, epochs: list, verbose: bool = False) -> list:
    """
    Automatically detect the ICA components that are not brain related and remove them.

    Arguments:
        icas: list of Independent Components for each participant (IC are MNE
          objects).
        epochs: list of 2 Epochs objects (for each participant). Epochs_S1
          and Epochs_S2 correspond to a condition and can result from the
          concatenation of Epochs from different experimental realisations
          of the condition.
          Epochs are MNE objects: data are stored in an array of shape
          (n_epochs, n_channels, n_times) and parameters information is
          stored in a disctionnary.
        verbose: option to plot data before and after ICA correction, 
          boolean, set to False by default. 

    Returns:
        cleaned_epochs_ICA: list of 2 cleaned Epochs for each participant
          (the non-brain related IC have been removed from the signal).
    """

    cleaned_epochs_ICA = []
    n_ICA_excluded = []
    for ica, epoch in zip(icas, epochs):
        ica_with_labels_fitted = label_components(epoch, ica, method="iclabel")
        print(ica_with_labels_fitted)
        ica_with_labels_component_detected = ica_with_labels_fitted["labels"]
        print(ica_with_labels_component_detected)
        # Remove non-brain components (take only brain components for each subject)
        excluded_idx_components = [idx for idx, label in enumerate(ica_with_labels_component_detected) if label not in ["brain"]]
        # take only eyes: 
        # excluded_idx_components = [idx for idx, label in enumerate(ica_with_labels_component_detected) if label in ["eye"]]
        print( excluded_idx_components)
        cleaned_epoch_ICA = mne.Epochs.copy(epoch)
        cleaned_epoch_ICA.info['bads'] = []
        ica.apply(cleaned_epoch_ICA, exclude=excluded_idx_components)
        cleaned_epoch_ICA.info['bads'] = copy.deepcopy(epoch.info['bads'])
        n_ICA_excluded.append(len(excluded_idx_components))
        cleaned_epochs_ICA.append(cleaned_epoch_ICA)

        if verbose:
            epoch.plot(title='Before ICA correction', show=True)
            cleaned_epoch_ICA.plot(title='After ICA correction',show=True)
    return cleaned_epochs_ICA, n_ICA_excluded


def AR_local_custom(cleaned_epochs_ICA: list, n_interpolates, consensus_percs, strategy:str = 'union', threshold:float = 50.0, verbose: bool = False) -> list:
    """
    Applies local Autoreject to repair or reject bad epochs.
    Arguments:
        clean_epochs_ICA: list of Epochs after global Autoreject and ICA.
        strategy: more or less generous strategy to reject bad epochs: 'union'
          or 'intersection'. 'union' rejects bad epochs from subject 1 and
          subject 2 immediatly, whereas 'intersection' rejects shared bad epochs
          between subjects, tries to repare remaining bad epochs per subject,
          reject the non-reparable per subject and finally equalize epochs number
          between subjects. Set to 'union' by default.
        threshold: percentage of epochs removed that is accepted. Above
          this threshold, data are considered as a too shortened sample
          for further analyses. Set to 50.0 by default.
        verbose: option to plot data before and after AR, boolean, set to
          False by default. # use verbose = false until next Autoreject update
    Note:
        To reject or repair epochs, parameters are more or less conservative,
        see http://autoreject.github.io/generated/autoreject.AutoReject.
    Returns:
        cleaned_epochs_AR: list of Epochs after local Autoreject.
        dic_AR: dictionnary with the percentage of epochs rejection
          for each subject and for the intersection of the them.
    """
    bad_epochs_AR = []
    AR = []
    dic_AR = {}
    dic_AR['strategy'] = strategy
    dic_AR['threshold'] = threshold

    # defaults values for n_interpolates and consensus_percs
    #n_interpolates = np.array([1, 4, 32])
    #consensus_percs = np.linspace(0, 1.0, 11)
    # more generous values
    # n_interpolates = np.array([16, 32, 64])
    # n_interpolates = np.array([1, 4, 8, 16, 32, 64])
    # consensus_percs = np.linspace(0.5, 1.0, 11)

    for clean_epochs in cleaned_epochs_ICA:  # per subj

        picks = mne.pick_types(
            clean_epochs[0].info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude=[])

        ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                        thresh_method='random_search', random_state=42,
                        verbose='tqdm_notebook')
        AR.append(ar)

        # fitting AR to get bad epochs
        ar.fit(clean_epochs)
        reject_log = ar.get_reject_log(clean_epochs, picks=picks)
        #ar.get_reject_log(clean_epochs).plot()
        #reject_log.plot('horizontal')
        #clean_epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
        bad_epochs_AR.append(reject_log)

    # taking bad epochs for min 1 subj (dyad)
    log1 = bad_epochs_AR[0]
    log2 = bad_epochs_AR[1]

    print(log1)

    bad1 = np.where(log1.bad_epochs == True)
    bad2 = np.where(log2.bad_epochs == True)

    print(bad1[0])

    if strategy == 'union':
        bad = list(set(bad1[0].tolist()).union(set(bad2[0].tolist())))
    elif strategy == 'intersection':
        bad = list(set(bad1[0].tolist()).intersection(set(bad2[0].tolist())))
    else:
        TypeError('not good strategy input!')

    # storing the percentage of epochs rejection
    dic_AR['S1'] = float((len(bad1[0].tolist())/len(cleaned_epochs_ICA[0]))*100)
    dic_AR['S2'] = float((len(bad2[0].tolist())/len(cleaned_epochs_ICA[1]))*100)

    # picking good epochs for the two subj
    cleaned_epochs_AR = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
        # keep a copy of the original data
        clean_epochs_ep = copy.deepcopy(clean_epochs)
        clean_epochs_ep = clean_epochs_ep.drop(indices=bad)
        # interpolating bads or removing epochs
        ar = AR[cleaned_epochs_ICA.index(clean_epochs)]
      #  print('len unprocessed epochs' + len(clean_epochs_AR))
        clean_epochs_AR = ar.transform(clean_epochs_ep)
     #   print('len epochs' + len(clean_epochs_AR))
        cleaned_epochs_AR.append(clean_epochs_AR)
        

        


    if strategy == 'intersection':
        # equalizing epochs length between two participants
        mne.epochs.equalize_epoch_counts(cleaned_epochs_AR)

    dic_AR['dyad'] = float(((len(cleaned_epochs_ICA[0])-len(cleaned_epochs_AR[0]))/len(cleaned_epochs_ICA[0]))*100)
    if dic_AR['dyad'] >= threshold:
        print('percentage of rejected epochs above threshold!')
        #TypeError('percentage of rejected epochs above threshold!')
    if verbose:
        print('%s percent of bad epochs' % dic_AR['dyad'])

    # Vizualisation before after AR
    evoked_before = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
        evoked_before.append(clean_epochs.average())

    evoked_after_AR = []
    for clean in cleaned_epochs_AR:
        evoked_after_AR.append(clean.average())

    if verbose:
        for i, j in zip(evoked_before, evoked_after_AR):
            fig, axes = plt.subplots(2, 1, figsize=(6, 6))
            for ax in axes:
                ax.tick_params(axis='x', which='both', bottom='off', top='off')
                ax.tick_params(axis='y', which='both', left='off', right='off')

            ylim = dict(grad=(-170, 200))
            i.pick_types(eeg=True, exclude=[])
            fig_before = i.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
            axes[0].set_title('Before autoreject')
            j.pick_types(eeg=True, exclude=[])
            fig_after = j.plot(exclude=[], axes=axes[1], ylim=ylim)
            # Problème titre ne s'affiche pas pour le deuxieme axe !!!
            axes[1].set_title('After autoreject')
            plt.tight_layout()


    return cleaned_epochs_AR, dic_AR, bad_epochs_AR




def AR_global_custom(cleaned_epochs_ICA: list, strategy:str = 'union', threshold:float = 50.0, verbose: bool = False) -> list:
    """
    Applies local Autoreject to repair or reject bad epochs.
    Arguments:
        clean_epochs_ICA: list of Epochs after global Autoreject and ICA.
        strategy: more or less generous strategy to reject bad epochs: 'union'
          or 'intersection'. 'union' rejects bad epochs from subject 1 and
          subject 2 immediatly, whereas 'intersection' rejects shared bad epochs
          between subjects, tries to repare remaining bad epochs per subject,
          reject the non-reparable per subject and finally equalize epochs number
          between subjects. Set to 'union' by default.
        threshold: percentage of epochs removed that is accepted. Above
          this threshold, data are considered as a too shortened sample
          for further analyses. Set to 50.0 by default.
        verbose: option to plot data before and after AR, boolean, set to
          False by default. # use verbose = false until next Autoreject update
    Note:
        To reject or repair epochs, parameters are more or less conservative,
        see http://autoreject.github.io/generated/autoreject.AutoReject.
    Returns:
        cleaned_epochs_AR: list of Epochs after local Autoreject.
        dic_AR: dictionnary with the percentage of epochs rejection
          for each subject and for the intersection of the them.
    """
    bad_epochs_AR = []
    AR = []
    dic_AR = {}
    dic_AR['strategy'] = strategy
    dic_AR['threshold'] = threshold

    # defaults values for n_interpolates and consensus_percs
    #n_interpolates = np.array([1, 4, 32])
    #consensus_percs = np.linspace(0, 1.0, 11)
    # more generous values
    # n_interpolates = np.array([16, 32, 64])
    # n_interpolates = np.array([1, 4, 8, 16, 32, 64])
    # consensus_percs = np.linspace(0.5, 1.0, 11)

    for clean_epochs in cleaned_epochs_ICA:  # per subj

        picks = mne.pick_types(
            clean_epochs[0].info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude=[])
            
        reject = get_rejection_threshold(clean_epochs)
        bad_epochs_AR.append(reject)

    # taking bad epochs for min 1 subj (dyad)

    cleaned_epochs_AR = []
    for participant in range(2):
        clean_epochs_AR = cleaned_epochs_ICA[participant].drop_bad(reject = bad_epochs_AR[participant])
    

    bad1 = np.nonzero(map(len, clean_epochs_AR[0].drop_log))
    bad2 = np.nonzero(map(len, clean_epochs_AR[1].drop_log))
    print(bad1)

    #bad1 = np.where(log1.bad_epochs == True)
    #bad2 = np.where(log2.bad_epochs == True)

    if strategy == 'union':
        bad = list(set(bad1.tolist()).union(set(bad2.tolist())))
    elif strategy == 'intersection':
        bad = list(set(bad1.tolist()).intersection(set(bad2.tolist())))
    else:
        TypeError('not good strategy input!')

    # storing the percentage of epochs rejection
    dic_AR['S1'] = float((len(bad1.tolist())/len(cleaned_epochs_ICA))*100)
    dic_AR['S2'] = float((len(bad2.tolist())/len(cleaned_epochs_ICA))*100)

    # picking good epochs for the two subj
    cleaned_epochs_AR = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
        # keep a copy of the original data
        clean_epochs_AR = copy.deepcopy(clean_epochs)
        clean_epochs_AR = clean_epochs_AR.drop(indices=bad)
       

    if strategy == 'intersection':
        # equalizing epochs length between two participants
        mne.epochs.equalize_epoch_counts(cleaned_epochs_AR)

    dic_AR['dyad'] = float(((len(cleaned_epochs_ICA[0])-len(cleaned_epochs_AR[0]))/len(cleaned_epochs_ICA[0]))*100)
    if dic_AR['dyad'] >= threshold:
        TypeError('percentage of rejected epochs above threshold!')
    if verbose:
        print('%s percent of bad epochs' % dic_AR['dyad'])


    return cleaned_epochs_AR, dic_AR, bad_epochs_AR


def custom_ICA_fit(epochs: list, n_components: int, method: str, fit_params: dict, random_state: int) -> list:
    """
    Computes global Autorejection to fit Independent Components Analysis
    on Epochs, for each participant.

    Pre requisite : install autoreject
    https://api.github.com/repos/autoreject/autoreject/zipball/master

    Arguments:
        epochs: list of 2 Epochs objects (for each participant).
          Epochs_S1 and Epochs_S2 correspond to a condition and can result
          from the concatenation of Epochs from different experimental
          realisations of the condition (Epochs are MNE objects).
        n_components: the number of principal components that are passed to the
          ICA algorithm during fitting, int. For a first estimation,
          n_components can be set to 15.
        method: the ICA method used, str 'fastica', 'infomax' or 'picard'.
          'Fastica' is the most frequently used. Use the fit_params argument to set
           additional parameters. Specifically, if you want Extended Infomax, set
           method=’infomax’ and fit_params=dict(extended=True) (this also works
           for method=’picard’). 
        fit_params: Additional parameters passed to the ICA estimator
           as specified by method. None by default.
        random_state: the parameter used to compute random distributions
          for ICA calulation, int or None. It can be useful to fix
          random_state value to have reproducible results. For 15
          components, random_state can be set to 97, for 20 components to 0
          for example.

    Note:
        If Autoreject and ICA take too much time, change the decim value
        (see MNE documentation).
        Please filter the Epochs between 2 and 30 Hz before ICA fit
        (mne.Epochs.filter(epoch, 2, 30, method='fir')).

    Returns:
        icas: list of Independant Components for each participant (IC are MNE
          objects, see MNE documentation for more details).
    """
    # do not use the autoreject part
    icas = []
    for epoch in epochs:
        # per subj
    

        # fitting ICA on filt_raw after AR
        ica = ICA(n_components=n_components,
                  method=method,
                  fit_params= fit_params,
                  random_state=random_state).fit(epoch)
        icas.append(ica.fit(epoch))

    return icas

    # Power of components
    p = wpwr(X - y)[0] / wpwr(X)[0]
    print('Power of components removed by DSS: {:.2f}'.format(p))
    # return the reconstructed clean signal, and the artifact
    return y, X - y


# helper function
def _multiply_conjugate(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Helper function to compute the product of a complex array and its conjugate.
    It is designed specifically to collapse the last dimension of a four-dimensional array.
    Arguments:
        real: the real part of the array.
        imag: the imaginary part of the array.
        transpose_axes: axes to transpose for matrix multiplication.
    Returns:
        product: the product of the array and its complex conjugate.
    """
    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


def _multiply_conjugate_time(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Helper function to compute the product of a complex array and its conjugate.
    Unlike _multiply_conjugate, this doenst collapse the last dimension of a 
    four-dimensional array. Useful when computing some connectivity metrics 
    (e.g., wpli), since it preserves the product values across e.g., time.
    
    Arguments:
        real: the real part of the array.
        imag: the imaginary part of the array.
        transpose_axes: axes to transpose for matrix multiplication.
    Returns:
        product: the product of the array and its complex conjugate.
    """
    formula = 'jilm,jimk->jilkm'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))
    
    return product


def compute_sync(complex_signal: np.ndarray, mode: str, epochs_average: bool = True, save_memory: bool = False, epochs_per_iteration: int = 1) -> np.ndarray:
    """
    Computes frequency- or time-frequency-domain connectivity measures from analytic signals.
    Arguments:
        complex_signal:
            shape = (2, n_epochs, n_channels, n_freq_bins, n_times).
            Analytic signals for computing connectivity between two participants.
        mode:
            Connectivity measure. Options in the notes.
        epochs_average:
            option to either return the average connectivity across epochs (collapse across time) or preserve epoch-by-epoch connectivity, boolean.
            If False, PSD won't be averaged over epochs (the time course is maintained).
            If True, PSD values are averaged over epochs.
        save_memory:
            option to create connectivity matrix epoch per epoch, rather than with all epochs at once.
            is slower but prevents running out of memory
    Returns:
        con:
            Connectivity matrix. The shape is either
            (n_freq, n_epochs, 2*n_channels, 2*n_channels) if time_resolved is False,
            or (n_freq, 2*n_channels, 2*n_channels) if time_resolved is True.
            To extract inter-brain connectivity values, slice the last two dimensions of con with [0:n_channels, n_channels: 2*n_channels].
    Note:
        **supported connectivity measures**
          - 'envelope_corr': envelope correlation
          - 'pow_corr': power correlation
          - 'plv': phase locking value
          - 'ccorr': circular correlation coefficient
          - 'coh': coherence
          - 'imaginary_coh': imaginary coherence
          - 'pli': phase lag index
          - 'wpli': weighted phase lag index
    """

    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    complex_signal_full = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
    transpose_axes = (0, 1, 3, 2)


    if save_memory:
        # loops through each epoch once
        epoch_ranges = range(n_epoch) 
        epochs_per_iteration = epochs_per_iteration

    else:
        # does one iteration that includes all epochs
        epoch_ranges = [range(n_epoch)]
        epochs_per_iteration = n_epoch


    for epoch_range in epoch_ranges:

        # take either full signal or one epoch
        complex_signal = complex_signal_full[epoch_range, :, :, :].reshape(epochs_per_iteration, n_freq, 2 * n_ch, n_samp)

        if mode.lower() == 'plv':
            phase = complex_signal / np.abs(complex_signal)
            c = np.real(phase)
            s = np.imag(phase)
            dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
            con = abs(dphi) / n_samp

        elif mode.lower() == 'envelope_corr':
            env = np.abs(complex_signal)
            mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
            env = env - mu_env
            con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
                np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

        elif mode.lower() == 'pow_corr':
            env = np.abs(complex_signal) ** 2
            mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
            env = env - mu_env
            con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
                np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

        elif mode.lower() == 'coh':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            amp = np.abs(complex_signal) ** 2
            dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
            con = np.abs(dphi) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                np.nansum(amp, axis=3)))

        elif mode.lower() == 'imaginary_coh':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            amp = np.abs(complex_signal) ** 2
            dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
            con = np.abs(np.imag(dphi)) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                            np.nansum(amp, axis=3)))

        elif mode.lower() == 'ccorr':
            angle = np.angle(complex_signal)
            mu_angle = circmean(angle, axis=3).reshape(epochs_per_iteration, n_freq, 2 * n_ch, 1) # used to be: .reshape(n_epochs, n_freq, 2 * n_ch, 1)
            angle = np.sin(angle - mu_angle)

            formula = 'nilm,nimk->nilk'
            con = np.einsum(formula, angle, angle.transpose(transpose_axes)) / \
                np.sqrt(np.einsum('nil,nik->nilk', np.sum(angle ** 2, axis=3), np.sum(angle ** 2, axis=3)))
            
        elif mode.lower() == 'pli':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            dphi = _multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
            con = abs(np.mean(np.sign(np.imag(dphi)), axis=4))
            
        elif mode.lower() == 'wpli':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            dphi = _multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
            con_num = abs(np.mean(abs(np.imag(dphi)) * np.sign(np.imag(dphi)), axis=4))
            con_den = np.mean(abs(np.imag(dphi)), axis=4)      
            con_den[con_den == 0] = 1 
            con = con_num / con_den        

        else:
            ValueError('Metric type not supported.')

        if save_memory:
            if epoch_range == 0:
                aggregate_con = con
            else:
                #con = np.expand_dims(con, axis = 0)
                aggregate_con = np.concatenate((aggregate_con, con), axis = 0)
        
    if save_memory:
        con = aggregate_con
    
    con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch
    if epochs_average:
        con = np.nanmean(con, axis=1)

    return con



def compute_freq_bands(data: np.ndarray, sampling_rate: int, freq_bands: dict, **filter_options) -> np.ndarray:
    """
    Computes analytic signal per frequency band using FIR filtering
    and Hilbert transform.

    Arguments:
        data:
            shape is (2, n_epochs, n_channels, n_times)
            real-valued data to compute analytic signal from.
        sampling_rate:
            sampling rate.
        freq_bands:
            a dictionary specifying frequency band labels and corresponding frequency ranges
            e.g. {'alpha':[8,12],'beta':[12,20]} indicates that computations are performed over two frequency bands: 8-12 Hz for the alpha band and 12-20 Hz for the beta band.
        enveloppe: 
            should we 
        **filter_options:
            additional arguments for mne.filter.filter_data, such as filter_length, l_trans_bandwidth, h_trans_bandwidth
    Returns:
        complex_signal: array, shape is
            (2, n_epochs, n_channels, n_freq_bands, n_times)
    """
    assert data[0].shape[0] == data[1].shape[0], "Two data streams should have the same number of trials."
    data = np.array(data)

    n_freq_bands = len(freq_bands.keys())
    n_epochs = data.shape[1]
    n_ch = data.shape[2]
    n_times = data.shape[3]
    # first transform the data so that you all epochs are concatenated
    #data = np.reshape(data, (2, n_ch, n_epochs * n_times))

    # filtering and hilbert transform
    complex_signal = []
    for freq_band in freq_bands.values():
        # first reshape the array so that the dimension of the epochs is merged with that of time
        # then use the length of the time dimension to take return it into epochs (use np.reshape) 
        
        print('filter_action')

        filtered = np.array([mne.filter.filter_data(data[participant],
                                                    sampling_rate, l_freq=freq_band[0], h_freq=freq_band[1],
                                                    **filter_options,
                                                    verbose=False)
                             for participant in range(2)
                             # for each participant
                             ])
        hilb = signal.hilbert(filtered)
        complex_signal.append(hilb)

    complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])

    # now split the data up back into the epochs
    #complex_signal = np.reshape(complex_signal, (2, n_epochs, n_ch, n_freq_bands, n_times))

    return complex_signal



def compute_freq_bands_memory_saving(data: np.ndarray, sampling_rate: int, freq_bands: dict, **filter_options) -> np.ndarray:
    """
    Computes analytic signal per frequency band using FIR filtering
    and Hilbert transform.

    Arguments:
        data:
            shape is (2, n_epochs, n_channels, n_times)
            real-valued data to compute analytic signal from.
        sampling_rate:
            sampling rate.
        freq_bands:
            a dictionary specifying frequency band labels and corresponding frequency ranges
            e.g. {'alpha':[8,12],'beta':[12,20]} indicates that computations are performed over two frequency bands: 8-12 Hz for the alpha band and 12-20 Hz for the beta band.
        enveloppe: 
            should we 
        **filter_options:
            additional arguments for mne.filter.filter_data, such as filter_length, l_trans_bandwidth, h_trans_bandwidth
    Returns:
        complex_signal: array, shape is
            (2, n_epochs, n_channels, n_freq_bands, n_times)
    """
    assert data[0].shape[0] == data[1].shape[0], "Two data streams should have the same number of trials."
    data = np.array(data)

    n_freq_bands = len(freq_bands.keys())
    n_epochs = data.shape[1]
    n_ch = data.shape[2]
    n_times = data.shape[3]
    # first transform the data so that you all epochs are concatenated
    data = np.reshape(data, (2, n_ch, n_epochs * n_times))

    segment_length = 50000  # Define a suitable segment length (number of time points)
    n_segments = (n_epochs * n_times) // segment_length  # Calculate the number of segments needed


    # filtering and hilbert transform
    complex_signal = []
    for freq_band in freq_bands.values():
        # first reshape the array so that the dimension of the epochs is merged with that of time
        # then use the length of the time dimension to take return it into epochs (use np.reshape) 
        processed_segments = []
    
        for segment_idx in range(n_segments + 1):  # +1 to include the last segment which might be shorter
            start_idx = segment_idx * segment_length
            end_idx = start_idx + segment_length
            if end_idx > n_epochs * n_times:
                end_idx = n_epochs * n_times  # Ensure we don't go beyond the data
            
            # Segment the data
            data_segment = data[:, :, start_idx:end_idx]
            
            # Filter and Hilbert transform the segmented data
            filtered_segment = np.array([
                mne.filter.filter_data(data_segment[participant], sampling_rate, 
                                    l_freq=freq_band[0], h_freq=freq_band[1],
                                    **filter_options, verbose=False)
                for participant in range(2)  # For each participant
            ])
            
            hilb_segment = signal.hilbert(filtered_segment, axis=-1)  # Apply the Hilbert transform along the time axis
            
            processed_segments.append(hilb_segment)

         # Concatenate the processed segments back together
    full_processed_signal = np.concatenate(processed_segments, axis=-1)
    complex_signal.append(full_processed_signal)

    complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])

    # now split the data up back into the epochs
    complex_signal = np.reshape(complex_signal, (2, n_epochs, n_ch, n_freq_bands, n_times))

    return complex_signal


def define_event_dictionary():
    # Define the possible values for each variable
    one_values = ['Synchronous', 'Complementary', 'Individual']
    two_values = ['Egalitarian', 'LeaderFollower', 'FollowerLeader']
    three_values = ['Success', 'Failure']
    four_values = ['Start', 'Checkpoint', 'Desyncpoint', 'Error']
    five_values = ['1', '2']

    # Create an empty dictionary to store the event IDs
    event_id = {}

    # Loop over all possible combinations of values for the five variables
    for one in one_values:
        for two in two_values:
            for three in three_values:
                for four in four_values:
                    for five in five_values:
                        # Create a key string using the values of the five variables
                        key = f"{one}/{two}/{three}/{four}/{five}"
                        
                        # Create an event ID using a unique integer value
                        event_id[key] = len(event_id) + 1

    return event_id


def create_erp_epochs(trials, sfreq, old_events, old_event_id, event_id):
    event_values = old_events[:, 2]
    event_samples = old_events[:,0]
    events = np.empty(3,)
    event_info = np.empty(3,) # (trial number, successful, baseline)
    # go through all the trials we found
    for trial_index in range(1, np.size(trials,0)):
        trial_start = trials[trial_index, 0]
        trial_end = trials[trial_index, 1]
        condition = trials[trial_index, 3]
        success = trials[trial_index, 4]
        condition_string = [k for k, v in old_event_id.items() if v == condition][0]
        if success == 1:
            success_string = 'Success'
        else:
            success_string = 'Failure'

        # now get the event at the start of the trial
        kind = 'Start'

        key = f"{condition_string}/{success_string}/{kind}/{'1'}"
        id = event_id[key]
        event = [trial_start, 0, id]
        events = np.vstack((events, np.array(event)))


        # find the events happening within this trial
        trial_events = np.where((trial_start < event_samples) & (trial_end > event_samples))[0]

        
        # conditions from which we want to have checkpoints
        if condition != 1:
            # find the first cross of the checkpoint
            crossed_points = []
            for trial_event in trial_events:
                event_value = str(event_values[trial_event])
                    
                if len(event_value) < 3:
                    continue
                event_sample = event_samples[trial_event]    
                player_crossed = int(event_value[0:2])
                checkpoint = int(event_value[2])
                event_sample = event_samples[trial_event]
                
                if ((player_crossed == 21) or (player_crossed == 11)) and (checkpoint not in crossed_points):
                    # add checkpoint to list
                    crossed_points.append(checkpoint)
                    
                    # only add the event when the last one was not less than 500ms before so as to not mix them too much
                    if (event_sample - events[-1, 0]) > sfreq:

                        first_crosser = event_value[0]
                        kind = 'Checkpoint'
                        key = f"{condition_string}/{success_string}/{kind}/{first_crosser}"
                        id = event_id[key]
                        event = [event_sample, 0, id]
                        events = np.vstack((events, np.array(event)))
                # or if they are game over
                elif ((player_crossed == 24) or (player_crossed == 14)):

                    first_crosser = event_value[0]
                    kind = 'Error'
                    key = f"{condition_string}/{success_string}/{kind}/{first_crosser}"
                    id = event_id[key]
                    event = [event_sample, 0, id]
                    events = np.vstack((events, np.array(event)))
                    break

            # find the first cross of the desync points
            crossed_points = []
            for trial_event in trial_events:
                event_value = str(event_values[trial_event])
                
                if len(event_value) < 3:
                    continue
                    
                player_crossed = int(event_value[0:2])
                checkpoint = int(event_value[2])

                # only for desync points
                if ((player_crossed == 20) or (player_crossed == 10)) and (checkpoint not in crossed_points):
                    # add desync point to list 
                    first_crosser = event_value[0]
                    kind = 'Desyncpoint'
                    key = f"{condition_string}/{success_string}/{kind}/{first_crosser}"
                    id = event_id[key]
                    event = [event_sample, 0, id]
                    events = np.vstack((events, np.array(event)))

                    crossed_points.append(checkpoint)
                          
    events = (np.rint(events)).astype(int)

    return events
