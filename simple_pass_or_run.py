import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt


DATA_PATH = "./nfl-big-data-bowl-2025"

# data = pd.read_csv(f"{DATA_PATH}/plays.csv")
# print(list(set(data['down'])))
# exit()


# things to add: yard number
# excessive movement before play:
# running: close to quarter back
# passing: switching

prediction_classes = ['quarter', 'down', 'yardsToGo', 'possessionTeam', 'defensiveTeam', 'playClockAtSnap']
prediction_categories = {
    'quarter': 'int_categorical',
    'down': 'int_categorical',
    'yardsToGo': 'int',
    'possessionTeam': 'team',
    'defensiveTeam': 'team',
    # 'gameClock': 'time',
    # 'offenseFormation': 'offenseFormation',
    'playClockAtSnap': 'int'
}

# data classes
teams = []
# can also do something like list(set(data['category'])) but might miss values
with open(f"teams.csv") as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        teams.append(row['Abbreviation'])

prediction_sizes = [5, 4, 1, len(teams), len(teams), 1]

formations = []
Id = []
X = []
Y = []

def get_one_hot(l, label):
    index = l.index(label)
    return [0] * index + [1] + [0] * (len(l) - index - 1)

categorical_mode = 'one_hot' # 'embeddings'
with open(f"{DATA_PATH}/plays.csv") as csvfile:
    data = csv.DictReader(csvfile)

    for row in data:
        # 0 is pass, 1 is run
        x = []
        for category in prediction_classes:
            format = prediction_categories[category]
            if format == 'int':
                value = 0 if row[category] == 'NA' else int(row[category])
                # so it is also a list like the other formats.
                x.append([value])
            elif format == 'int_categorical':
                encoding = None
                if category == 'quarter':
                    if categorical_mode == 'one_hot':
                        encoding = get_one_hot([1, 2, 3, 4, 5], int(row[category]))
                    elif categorical_mode == 'embedding':
                        pass
                elif category == 'down':
                    if categorical_mode == 'one_hot':
                        encoding = get_one_hot([1, 2, 3, 4], int(row[category]))
                    elif categorical_mode == 'embedding':
                        pass
                else:
                    print(f'error: no class {category} matched for int_categorical')
                
                x.append(encoding)
                
            elif format == 'team':
                one_hot = get_one_hot(teams, row[category])
                x.append(one_hot)
            
        y = 0 if row['passResult'] else 1

        Id.append(int(row["playId"]))
        X.append(x)
        Y.append(y)

class Model(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=50):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.norm_1 = nn.BatchNorm1d(hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.norm_2 = nn.BatchNorm1d(hidden_size)
        self.layer_3 = nn.Linear(hidden_size, output_size)
        self.norm_3 = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.norm_1(x)
        x = F.sigmoid(x)

        x = self.layer_2(x)
        x = self.norm_2(x)
        x = F.sigmoid(x)

        x = self.layer_3(x)
        x = self.norm_3(x)
        x = F.sigmoid(x)
        return x
    
input_size = sum(prediction_sizes)
# for prediction_size in prediction_sizes:
#     input_size += prediction_size
print(f'input_size {input_size}')

# model
model = Model(input_size)

# loss and optimizer
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

length = len(X)
training_length = length // 2
testing_length = (length // 2 + (length % 2 == 1)) - training_length // 4
validation_length = training_length // 4
print(f'Training_length: {training_length}')
print(f'Testing_length: {testing_length}')
print(f'Validation_length: {validation_length}')

trainX = X[:training_length]
trainY = Y[:training_length]

testX = X[training_length:training_length + testing_length]
testY = Y[training_length:training_length + testing_length]

validationX = X[training_length + testing_length:]
validationY = Y[training_length + testing_length:]

# train
model.train()
num_epochs = 30
batch_size = 128
epoch_rec = []
train_loss_rec = []
validation_loss_rec = []
validation_scale = 50

for epoch in range(1, num_epochs):
    print(f'Epoch: {epoch}/{num_epochs}')
    batch_loss = 0
    for batch in range(training_length // batch_size):
        x_unconcatenated = trainX[batch * batch_size: (batch + 1) * batch_size]
        y = trainY[batch * batch_size: (batch + 1) * batch_size]
        x = []
        for play in x_unconcatenated:
            sample = []
            for l in play:
                for e in l:
                    sample.append(e)
            x.append(sample)
        x = torch.Tensor(x)
        # print(f'Input shape: {x.shape}')
        y = torch.Tensor(y)

        pred_y = model(x).view(-1)

        # print(f'pred shape: {pred_y.shape}')
        # print(f'target shape: {y.shape}')
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f'pred_y: {pred_y[:10]}')
        # print(f'y: {y[:10]}')
        # print(f'Loss: {loss}')
        batch_loss += loss

    # validation
    with torch.no_grad():
        y = validationY
        x = []
        for play in validationX:
            sample = []
            for l in play:
                for e in l:
                    sample.append(e)
            x.append(sample)
        x = torch.Tensor(x)
        y = torch.Tensor(y)

        pred_y = model(x).view(-1)
        loss = criterion(pred_y, y) * validation_scale

    epoch_rec.append(epoch)
    train_loss_rec.append(batch_loss)
    validation_loss_rec.append(loss)
    

with torch.no_grad():
    y = testY
    x = []
    for play in testX:
        sample = []
        for l in play:
            for e in l:
                sample.append(e)
        x.append(sample)
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    pred_y = model(x).view(-1)
    pred_closest_y = torch.round(pred_y)
    loss = criterion(pred_y, y)

    print(f'Pred: {pred_y[:50]}')
    print(f'Closest Pred:\n{pred_closest_y[:50]}')
    print(f'Target:\n{y[:50]}')
    print(f'Loss: {loss}')
    print(f'Loss closest: {criterion(pred_closest_y, y)}')

    plt.title('Loss over epochs')
    plt.plot(epoch_rec, train_loss_rec, label='Training Loss')
    plt.plot(epoch_rec, validation_loss_rec, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()