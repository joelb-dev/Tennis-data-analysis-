#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
# load and investigate the data here:
file_path = r"C:\Users\Joel\Downloads\tennis_ace_starting\tennis_ace_starting\tennis_stats.csv"
df = pd.read_csv(file_path)

#offensive plays
player = df["Player"]
year = df["Year"]
aces = df["Aces"]
double_faults = df["DoubleFaults"]
first_serve = df["FirstServe"]
first_serve_points_won = df["FirstServePointsWon"]
second_serve_points_won = df["SecondServePointsWon"]
break_points_faced = df["BreakPointsFaced"]
break_points_saved = df["BreakPointsSaved"]
service_games_played = df["ServiceGamesPlayed"]
service_games_won = df["ServiceGamesWon"]
total_service_points_won = df["TotalServicePointsWon"]
#defensive plays
first_serve_return_points_won = df["FirstServeReturnPointsWon"]
second_serve_return_points_won = df["SecondServeReturnPointsWon"]
break_points_opportunities = df["BreakPointsOpportunities"]
break_points_converted = df["BreakPointsConverted"]
return_games_played = df["ReturnGamesPlayed"]
return_games_won = df["ReturnGamesWon"]
return_points_won = df["ReturnPointsWon"]
total_points_won = df["TotalPointsWon"]
#outcomes
wins = df["Wins"]
losses = df["Losses"]
winnings = df["Winnings"]
ranking = df["Ranking"]
# perform exploratory analysis here:
#plt.scatter(break_points_opportunities, winnings)

## perform single feature linear regressions here:
features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = df[['Winnings']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

model.score(features_test, outcome_test)

prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.show()















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
