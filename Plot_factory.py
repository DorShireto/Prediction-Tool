import pandas as pd
import matplotlib.pyplot as plt

def home_away_top_players_plot():
    # full_features = pd.read_csv('./player_top_75/full_features.csv')
    full_features = pd.read_csv('full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['home_top_players'], full_features['away_top_players'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    # snc.regplot('home_top_players','away_top_players',data=full_features,fit_reg=True)
    plt.xlabel("home_top_players")
    plt.ylabel("away_top_players")
    plt.title("top_players above 80")
    plt.show()

def home_away_mean_rating_plot():
    full_features = pd.read_csv('full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['home_team_mean_rating'], full_features['away_team_mean_rating'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("home_team_mean_rating")
    plt.ylabel("away_team_mean_rating")
    plt.title("mean_rating for home and away team")
    plt.show()

def home_away_5_games_pts_plot():
    full_features = pd.read_csv('full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['h_5_games_pts'], full_features['a_5_games_pts'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("h_5_games_pts")
    plt.ylabel("a_5_games_pts")
    plt.title("5_games_pts for home and away team")
    plt.show()

def Away_and_Win_best_plot():
    full_features = pd.read_csv('./player_top_81/full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['AVG_bet_H'], full_features['AVG_bet_A'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("AVG_bet_H")
    plt.ylabel("AVG_bet_A")
    plt.title("AVG_bet for Away and Home teams")
    plt.show()

def Draw_and_Win_best_plot():
    full_features = pd.read_csv('full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['AVG_bet_H'], full_features['AVG_bet_A_D'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("AVG_bet_H")
    plt.ylabel("AVG_bet_A_D")
    plt.title("AVG_bet for home team win or away team win/draw")
    plt.show()

def buildUp_speed_plot():
    full_features = pd.read_csv('./with_team_att/full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['buildUpPlaySpeedHome'], full_features['buildUpPlaySpeedAway'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("buildUpPlaySpeedHome")
    plt.ylabel("buildUpPlaySpeedAway")
    plt.title("buildUpPlaySpeed for Home and Away team")
    plt.show()

def defencePressure_plot():
    full_features = pd.read_csv('./with_team_att/full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['defencePressureHome'], full_features['defencePressureAway'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("defencePressureHome")
    plt.ylabel("defencePressureAway")
    plt.title("defencePressure for Home and Away team")
    plt.show()

def defenceAggression_plot():
    full_features = pd.read_csv('./with_team_att/full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['defenceAggressionHome'], full_features['defenceAggressionAway'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("defenceAggressionHome")
    plt.ylabel("defenceAggressionAway")
    plt.title("defenceAggression for Home and Away team")
    plt.show()

def chanceCreationPassing_plot():
    full_features = pd.read_csv('./with_team_att/full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['chanceCreationPassingHome'], full_features['chanceCreationPassingAway'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("chanceCreationPassingHome")
    plt.ylabel("chanceCreationPassingAway")
    plt.title("chanceCreationPassing for Home and Away team")
    plt.show()

def chanceCreationShooting_plot():
    full_features = pd.read_csv('./with_team_att/full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['chanceCreationShootingHome'], full_features['chanceCreationShootingAway'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("chanceCreationShootingHome")
    plt.ylabel("chanceCreationShootingAway")
    plt.title("chanceCreationShooting for Home and Away team")
    plt.show()



def home_away_only_x_games_pts_plot(x):
    full_features = pd.read_csv('full_features.csv')
    fig, ax = plt.subplots()
    # 0 - lose and draw home team
    # 1 - win to home team
    colors = {0: 'red', 1: 'blue'}
    ax.scatter(full_features['h_only_'+str(x)+'_games_pts'], full_features['a_only_'+str(x)+'_games_pts'],
               c=full_features['final_result'].apply(lambda x: colors[x]), alpha=0.5)
    plt.xlabel("h_only_"+str(x)+"_games_pts")
    plt.ylabel("a_only_"+str(x)+"_games_pts")
    plt.title(str(x)+"_games_pts for home only and away only team")
    plt.show()


####### main ######
# home_away_top_players_plot()
# home_away_mean_rating_plot()
# home_away_5_games_pts_plot()
# Away_and_Win_best_plot()
# Draw_and_Win_best_plot()
# buildUp_speed_plot()
# defencePressure_plot()
# defenceAggression_plot()
# chanceCreationPassing_plot()
# chanceCreationShooting_plot()
# home_away_5_games_pts_plot()
# home_away_only_x_games_pts_plot(1)
# home_away_only_x_games_pts_plot(2)
# home_away_only_x_games_pts_plot(3)
# home_away_only_x_games_pts_plot(4)
# home_away_only_x_games_pts_plot(5)
# home_away_only_x_games_pts_plot(10)
