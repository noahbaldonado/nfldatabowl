import pandas as pd

DATA_PATH = "./nfl-big-data-bowl-2025"

data = pd.read_csv(f"{DATA_PATH}/tracking_week_1.csv")
data.fillna(-1, inplace=True)
# print(data)

# get games
gameIds = data['gameId'].unique()
gameId = gameIds[5]
game = data[data['gameId'] == gameId]

# get playIds
playIds = game['playId'].unique()

# start looking at the play with the lowest playId
playId = playIds[5]
play = game[game['playId'] == playId]

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

club_a = -1

events = {}

for time in times:
    moment = play[play['time'] == time]
    for i, row in moment.iterrows():
        if (clubs[row['nflId']] == -1):
            clubs[row['nflId']] = row['club']
            if club_a == -1:
                club_a = row['club'];
        
        # if len(locations[row['nflId']]) > 10:
        #     continue
        locations[row['nflId']].append((row['x'], row['y']))
        
        events[time] = row['event']


# template from Google AI
import pygame

# Initialize Pygame
pygame.init()

# Set screen dimensions
screen_width = 1000
screen_height = 1000
screen = pygame.display.set_mode((screen_width, screen_height))

# Set title
pygame.display.set_caption("Play")

# Game loop
running = True
i = 0

# fps
clock = pygame.time.Clock()
# each frame is 0.03 seconds
fps = 1 / 0.06

screen.fill((0, 0, 0))  # Black background
font = pygame.font.SysFont("Arial", 36)
text = 'None'
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if i >= len(times) or not running:
        break

    # Clear the screen
    screen.fill((0, 0, 0))  # Black background

    time = times[i]
    print(events[time])
    if events[time] != -1:
        text = events[time]
    txtsurf = font.render(f'Recent Event: {text}', True, 'white')
    screen.blit(txtsurf,(0, 0))

    # Draw a red rectangle
    for player in players:
        location = (locations[player][i][0] * 10, locations[player][i][1] * 10)
        # print(location)
        if clubs[player] == 'football':
            pygame.draw.circle(screen, color='yellow', center=location, radius=5)
        elif clubs[player] == club_a:
            pygame.draw.circle(screen, color='red', center=location, radius=5)
        else:
            pygame.draw.circle(screen, color='blue', center=location, radius=5)

    # Update the display
    pygame.display.flip()
    i += 1

    clock.tick(fps)

# Quit Pygame
pygame.quit()