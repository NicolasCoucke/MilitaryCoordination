import numpy as np

class TrialClass:


    def __init__(self):
        self.TrialNumber = 0
        self.Condition = 0
        self.Success = False
        self.CompletionTime = 0
        self.time = []
        self.Player_1_x = []
        self.Player_1_y = []
        self.Player_2_x = []
        self.Player_2_y = []



    def get_game_time(self, cell):
        self.time.append(float(cell))

    def get_next_cell(self, string, begin_comma_index):
        # finds the next comma in line
        end_comma_index = string.find(",", begin_comma_index + 1)

        cell = string[begin_comma_index + 1:end_comma_index]
        return cell, end_comma_index


    def get_position_1(self, string, start_position):
        # print(player.PlayerNumber)
        position = string[string.find("(", start_position) + 1:string.find(")", start_position)]
        posindex = position.find(",")
        Xpos = float(position[0:posindex])
        Ypos = float(position[posindex + 1:])
        self.Player_1_x.append(Xpos)
        self.Player_1_y.append(Ypos)
        end_position = string.find(")", start_position + 1) + 1
        return end_position

    def get_position_2(self, string, start_position):
        # print(player.PlayerNumber)
        position = string[string.find("(", start_position) + 1:string.find(")", start_position)]
        posindex = position.find(",")
        Xpos = float(position[0:posindex])
        Ypos = float(position[posindex + 1:])
        self.Player_2_x.append(Xpos)
        self.Player_2_y.append(Ypos)
        end_position = string.find(")", start_position + 1) + 1
        return end_position




    trial = 0

    time = []
    Player_1_x = []
    Player_1_y = []

    Player_2_x = []
    Player_2_y = []


    def ParseTrial(self, data, startindex):
        # , error_bad_lines=False)
        # df = pd.read_csv("2020_11_24-13_14_19-8p-crown.csv", sep='delimiter', header=None, engine='python')#, error_bad_lines=False)
        #print(data.size)
        # data = df.to_numpy()
        index = startindex
        # print(float(data[index][0]))
        GameOverVar = 0
        FinishedVar = 0
        Overcounter = 0
        condition = []
        string = data[index]
        self.Condition = string.split(" ")[0]
        self.TrialNumber = string.split(" ")[1].split(",")[0]
        index+=1
        while (index < data.size):  # float(data[index+1][0]) >= float(data[index][0])):
            string = data[index]
            if "GameOver" in string:
                self.Success = False
                self.CompletionTime = max(self.time)
                while ("ync" not in string and index+1 < data.size): #move until next trial
                    index += 1
                    string = data[index]
                return index
            elif "Finished" in string:
                self.Success = True
                self.CompletionTime = max(self.time)
                while("ync" not in string and index+1 < data.size): #make sure it's a real succes and not one that precedes a failure
                    index+=1
                    string = data[index]
                    if "GameOver" in string:
                        self.Success = False
                return index
            else:
                strindex = 0
                strindex = string.find(",", 0)
                # fix here (crashes because only a zero in the file)
                cell = string[0:strindex]
                self.get_game_time(cell)
                strindex = self.get_position_1(string, strindex)
                strindex = self.get_position_2(string, strindex)
                index += 1
        return index









