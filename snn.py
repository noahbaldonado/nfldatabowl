import pandas as pd
import random
import numpy as np

NUM_PLAYS = 5


DATA_PATH = "./nfl-big-data-bowl-2025"

# all training data is from tracking_week_1
tracking_data = pd.read_csv(f"{DATA_PATH}/tracking_week_1.csv")
tracking_data.fillna(-1, inplace=True)

plays_data = pd.read_csv(f"{DATA_PATH}/plays.csv")

# sequence of frames
train_X = []
# yards gained
train_y = []

# get games
gameIds = tracking_data['gameId'].unique()

for play_number in range(NUM_PLAYS):
    # choose a random game
    gameId = gameIds[random.randint(0, len(gameIds) - 1)]
    game = tracking_data[tracking_data['gameId'] == gameId]

    # get playIds
    playIds = game['playId'].unique()

    # choose a random play
    playId = playIds[random.randint(0, len(playIds) - 1)]
    play = game[game['playId'] == playId]

    # from plays.csv
    play_data = plays_data[plays_data['playId'] == playId]

    # get players (maybe should use displayName?)
    players = play['nflId'].unique()
    num_players = len(players)

    # get clubs
    clubs = play['club'].unique()

    # get times
    # I think it is already sorted
    times = play['time'].unique()

    locations = {}
    for player in players:
        locations[player] = []

    clubs = {}
    for player in players:
        clubs[player] = -1

    possession_team = play_data['possessionTeam']
    defensive_team = play_data['defensiveTeam']

    events = {}
    
    # convert data into frames
    frames = []
    done = False
    for time in times:
        # 3 channels, 360 length (ft), 160 width (ft)
        # possesive team, other team, football
        frame = np.zeros((3, 360, 160))

        moment = play[play['time'] == time]
        for i, row in moment.iterrows():
            if row['frameType'] == 'SNAP':
                done = True
                break
            events[time] = row['event']
            x = int(row['x'])
            y = int(row['y'])
            if (clubs[row['nflId']] == -1):
                frame[2, x, y] = 1
            elif (clubs[row['nflId']] == possession_team):
                frame[0, x, y] += 1
            else:
                frame[1, x, y] += 1
        if done:
            break
        frames.append(frame)

    train_X.append(frames)
    print('dsfkgmsg')
    print(play_data['yardsGained'])
    print(type(play_data['yardsGained']))
    exit()
    train_y.append(int(play_data['yardsGained']))

print(f'X: {train_X}')
print(f'y: {train_y}')
            
    

