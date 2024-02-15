from Trial import TrialClass

def __init__(self, data):
    self.Data = data

def get_game_time(cell):
    time.append(float(cell))

def get_next_cell(begin_comma_index):
    # finds the next comma in line
    end_comma_index = string.find(",", begin_comma_index + 1)

    cell = string[begin_comma_index + 1:end_comma_index]
    return cell, end_comma_index

def get_ping(cell, player):
    # print(player.PlayerNumber)
    ping = int(cell)
    player.AppendPing(ping)

def get_position_1(start_position):
    # print(player.PlayerNumber)
    position = string[string.find("(", start_position) + 1:string.find(")", start_position)]
    posindex = position.find(",")
    Xpos = float(position[0:posindex])
    Ypos = float(position[posindex + 1:])
    Player_1_x.append(Xpos)
    Player_1_y.append(Ypos)
    end_position = string.find(")", start_position + 1) + 1
    return end_position

def get_position_2(start_position):
    # print(player.PlayerNumber)
    position = string[string.find("(", start_position) + 1:string.find(")", start_position)]
    posindex = position.find(",")
    Xpos = float(position[0:posindex])
    Ypos = float(position[posindex + 1:])
    Player_2_x.append(Xpos)
    Player_2_y.append(Ypos)
    end_position = string.find(")", start_position + 1) + 1
    return end_position

MilitaryList = []
CivilianList = []
ConditionList = []
TrialList = []
FailureList = []
trial = 0

time = []
Player_1_x = []
Player_1_y = []

Player_2_x = []
Player_2_y = []


def Extract_Game_Data(self):



    # , error_bad_lines=False)
    # df = pd.read_csv("2020_11_24-13_14_19-8p-crown.csv", sep='delimiter', header=None, engine='python')#, error_bad_lines=False)
    data = self.Data[0]
    print(data.size)
    # data = df.to_numpy()
    index = 1
    # print(float(data[index][0]))
    GameOverVar = 0
    FinishedVar = 0
    Overcounter = 0
    condition = []
    while (index < data.size):  # float(data[index+1][0]) >= float(data[index][0])):
        string = data[index]
        if GameOverVar == 1 or FinishedVar == 1:
            if "ync" in string:
                condition = string
                GameOverVar = 0
                FinishedVar = 0
                print("newcondition")

        else:
            if "GameOver" in string:
                GameOverVar = 1
                time = []
                Player_1_x = []
                Player_1_y = []
                Player_2_x = []
                Player_2_y = []
                Overcounter += 1
            elif "Finished" in string:
                FinishedVar = 1
                Trial = TrialClass(trial, condition, time, Player_1_x, Player_1_y, Player_2_x, Player_2_y)
                #TrialList.append(Trial)
                #trial += 1
                time = []
                Player_1_x = []
                Player_1_y = []
                Player_2_x = []
                Player_2_y = []
            else:
                strindex = 0
                strindex = string.find(",", 0)
                # fix here (crashes because only a zero in the file)
                cell = string[0:strindex]
                get_game_time(cell)
                strindex = get_position_1(strindex)
                strindex = get_position_2(strindex)
        index += 1
    print(Overcounter)
    print(TrialList)





