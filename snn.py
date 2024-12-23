# data processing

import pandas as pd
import random
import numpy as np
import math

NUM_PLAYS = 100

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

# function for filling in a frame at a point
def fill(frame, c, x, y, r=10):
    for i in range(-r, r + 1):
        height = int(math.sqrt(r**2 - i**2))
        for j in range(-height, height + 1):
            distance = math.sqrt(i**2 + j**2)
            strength = ((r - distance) / r) ** 2
            if 0 < x + i < frame.shape[1] and 0 < y + j < frame.shape[2]:
                frame[c, x + i, y + j] += 10 * strength
            # print(f'Strenght: {strength}')

# 16 games, ~100-150 plays per game

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
    play_data = plays_data[(plays_data['gameId'] == gameId) & (plays_data['playId'] == playId)].iloc[0]

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


    possession_team = play_data['possessionTeam']
    defensive_team = play_data['defensiveTeam']

    events = {}
    
    # convert data into frames
    frames = []
    done = False
    for time in times:
        # 3 channels, 360 length (ft), 160 width (ft)
        # 3 channels = possesion team, other team, football
        frame = np.zeros((3, 360, 160))

        moment = play[play['time'] == time]
        for _, row in moment.iterrows():
            if row['frameType'] == 'SNAP':
                done = True
                break
            events[time] = row['event']
            x = int(row['x'])
            y = int(row['y'])
            if (row['club'] == possession_team):
                fill(frame, 0, x, y)
                # frame[0, x, y] += 1
            elif (row['club'] == defensive_team):
                fill(frame, 1, x, y)
                # frame[1, x, y] += 1
            else:
                fill(frame, 2, x, y)
                # frame[2, x, y] += 1 # football
        if done:
            break
        frames.append(frame)

    train_X.append(frames)
    # train_y.append(int(play_data['yardsGained']))
    train_y.append(max(0, int(play_data['yardsGained'])))

# train_X: list, list, ndarray of shape (3, 360, 160)
# snntorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

class SpkNet(nn.Module):

    def __init__(self, fc_input_size, hidden_size, output_size, beta, spike_grad):
        super().__init__()

        # Convolutions
        self.pool = nn.MaxPool2d(2, 2)

        # (batch_size, 3, 360, 160)
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.lifc1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.lifc2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.lifc3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc_input_size = fc_input_size

        # FC
        self.lin1 = nn.Linear(self.fc_input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lin4 = nn.Linear(hidden_size, output_size)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def display_conv(self, data, batch_size, c1, c2, x, y):
        for b in range(batch_size):
            data = np.zeros(c1 * c2, )
            for i in range(c1):
                for j in range(c2):
                    data[c1 * i + j
                    # working on this right now

    def forward(self, x):
        spk_rec = []
        # Number of frames
        num_steps = x.shape[0]

        mem_c1 = self.lifc1.init_leaky()
        mem_c2 = self.lifc2.init_leaky()
        mem_c3 = self.lifc3.init_leaky()
        mem_1 = self.lif1.init_leaky()
        mem_2 = self.lif2.init_leaky()
        mem_3 = self.lif3.init_leaky()
        mem_4 = self.lif4.init_leaky()

        print(f'/{num_steps}')
        for step in range(num_steps):
            if (step + 1) % 5 == 0:
                print(step + 1, end=', ')

            # Convolutions
            # It performs the convolution on the current frame, then pools.
            # There is no ReLU activation function needed, because it uses spikes!
            out = self.conv1(x[step])
            out = self.pool(out)
            spk_c1, mem_c1 = self.lifc1(out, mem_c1)
            out = self.conv2(spk_c1)
            out = self.pool(out)
            spk_c2, mem_c2 = self.lifc2(out, mem_c2)
            out = self.conv3(spk_c2)
            out = self.pool(out)
            spk_c3, mem_c3 = self.lifc3(out, mem_c3)

            # FC
            out = spk_c3.view(-1, self.fc_input_size) # Flatten
            out = self.lin1(out)
            spk_1, mem_1 = self.lif1(out, mem_1)
            out = self.lin2(spk_1)
            spk_2, mem_2 = self.lif2(out, mem_2)
            out = self.lin3(spk_2)
            spk_3, mem_3 = self.lif2(out, mem_3)
            out = self.lin4(spk_3)
            spk_4, mem_4 = self.lif4(out, mem_4)
            # Keep track of the spikes from the last layer.
            if num_steps - step <= 100:
                spk_rec.append(spk_4)

        # Return the record of the spikes
        return torch.stack(spk_rec, dim=0)
        

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.9

# model
fc_input_size = 64 * 41 * 16 # 41984
hidden_size = 1
output_size = 100
model = SpkNet(fc_input_size, hidden_size, output_size, beta, spike_grad)

# Learning rate
learning_rate = 0.01
target_correct = 0.8

# loss and optimizer
# to improve, take weighted average of spikes to find prediction
criterion = SF.mse_count_loss(correct_rate=target_correct, incorrect_rate=1-target_correct)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

num_samples = len(train_X)
print(f'num_samples: {num_samples}')
num_epochs = 100
batch_size = 5

for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    for i in range(0, num_samples, batch_size):
        data_list = train_X[i * batch_size: (i + 1) * batch_size]
        targets_list = train_y[i * batch_size: (i + 1) * batch_size]
        if (len(data_list) < batch_size):
            break

        num_frames = 100 # min
        for blah in data_list:
            if len(blah) > num_frames:
                num_frames = len(blah)
        print(f'num_frames: {num_frames}')

        frame_shape = data_list[0][0].shape
        data_numpy_list = []
        for blah in data_list:
            pad = np.zeros((num_frames - len(blah), *frame_shape))
            data_numpy = np.concatenate((pad, np.array(blah)))
            data_numpy_list.append(data_numpy)
        
        data = torch.from_numpy(np.array(data_numpy_list, dtype=np.float32))
        data = torch.permute(data, (1, 0, 2, 3, 4))
        targets = torch.from_numpy(np.array(targets_list, dtype=np.float32))
        # print(data.sum(dim=(0, 2, 3, 4)))
        # print(data.shape)
        # exit()
        spk_rec = model(data)   
        
        optimizer.zero_grad()
        loss = 0
        for b in range(batch_size):
            target = int(targets_list[b])
            s = spk_rec[:, b, :]
            for j in range(output_size):
                loss += s[j].sum() * min(1, (((j - 20) - target) ** 2 - 20)) / batch_size       
        # criterion = SF.mse_count_loss(correct_rate=target_correct, incorrect_rate=1-target_correct)
        # loss = criterion(spk_rec, targets)   

        loss.backward()
        optimizer.step()
        utils.reset(model)
        sums = spk_rec.sum(dim=0)
        predictions = sums.argmax(dim=1) - 20
        # print(sums)
        # print(loss)
        print(f'\nPredictions: {predictions}')
        print(f'Targets: {targets}')
        



