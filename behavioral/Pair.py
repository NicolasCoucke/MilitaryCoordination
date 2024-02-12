from Trial import TrialClass
import numpy as np
class PairClass:

    def __init__(self, pairnumber):
        self.Pair = pairnumber
        self.Trial_conditions = []
        self.Trial_numbers = []
        self.Trial_successes = []
        self.Trial_completion_times = []
        self.TrialList = []


    #deal with trials that first show "finished" and then gameover
    #if gameover before next trial begins than trial is a failure!!

    def ParsePair(self, data):
        # questionnaires
        index = 0
       # print(index)
        while (index < data.size):  # float(data[index+1][0]) >= float(data[index][0])):
            string = data[index]
            if "ync" in string:
                condition = string
                GameOverVar = 0
                FinishedVar = 0
                trial = TrialClass()
                index = trial.ParseTrial(data, index)
                self.TrialList.append(trial)
            else:
                index+=1
            #print(index)
        self.Generate_trial_info(self.TrialList)



    def Generate_trial_info(self, TrialList):
        self.Trial_info = np.zeros((len(self.TrialList),4))
        t=0
        for trial in TrialList:
            self.Trial_conditions.append(trial.Condition)
            self.Trial_numbers.append(trial.TrialNumber)
            self.Trial_successes.append(trial.Success)
            self.Trial_completion_times.append(trial.CompletionTime)




