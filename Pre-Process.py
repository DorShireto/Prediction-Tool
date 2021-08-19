import sqlite3
import pandas as pd
import numpy as np
import datetime


def get_closest_date_to_match(match_date, other_dates):
    # return date from other_dates,that is the closest to given (match_date)
    if len(other_dates) == 0:
        return 0
    elif len(other_dates) == 1:
        return other_dates
    else:
        # convert match_date from str type to Timestamp
        match_date = datetime.datetime.strptime(match_date, '%Y-%m-%d %H:%M:%S')

        diff = [abs(match_date - i) for i in other_dates]
        closest_date = min(diff)
        return other_dates[diff.index(closest_date)]


def get_last_x_home_matches(matches, date, team, x):
    # return the last X home games that the team have played before the date giving
    team_matches = matches[(matches['home_team_api_id'] == team)]
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]
    return last_matches


def get_last_x_away_matches(matches, date, team, x):
    # return the last X away games that the team have played before the date giving
    team_matches = matches[(matches['away_team_api_id'] == team)]
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]
    return last_matches


def get_last_x_matches_score(matches, team_id):
    '''
    :param matches:
    :return: score team earned in the last matches that giving the the matches DF
    '''
    score_in_x_matches = 0
    for index, row in matches.iterrows():
        home_team = row['home_team_api_id']
        away_team = row['away_team_api_id']
        home_team_goals = row['home_team_goal']
        away_team_goals = row['away_team_goal']
        # final_result = row['final_result']
        if home_team_goals == away_team_goals:
            score_in_x_matches += 1
        elif (home_team_goals > away_team_goals and home_team == team_id) or (
                home_team_goals < away_team_goals and away_team == team_id):
            score_in_x_matches += 3
    return score_in_x_matches


def add_last_x_matches_score(cleared_matches, x):
    """
    This method will add 2 new columns to the cleared_matches DF with the amount of point
    each team earned in their last home (for home team) or away (for away team) games
    :param cleared_matches: - DF with all the matches
    :param x: - The amount of games the user wants to get
    :return: -
    """
    last_x_games_score_home = []
    last_x_games_score_away = []

    for index, match in cleared_matches.iterrows():
        home_team = match['home_team_api_id']
        away_team = match['away_team_api_id']
        match_date = match['date']

        home_team_last_x_games = get_last_x_home_matches(cleared_matches, match_date, home_team, x)
        away_team_last_x_games = get_last_x_away_matches(cleared_matches, match_date, away_team, x)

        home_team_last_x_score = get_last_x_matches_score(home_team_last_x_games, home_team)
        away_team_last_x_score = get_last_x_matches_score(away_team_last_x_games, away_team)

        last_x_games_score_home.append(home_team_last_x_score)
        last_x_games_score_away.append(away_team_last_x_score)

    home_label = 'h_only_' + str(x) + '_games_pts'
    away_label = 'a_only_' + str(x) + '_games_pts'
    cleared_matches[home_label] = last_x_games_score_home
    cleared_matches[away_label] = last_x_games_score_away


def get_last_5_matches(matches, date, team):
    """
    :param matches: DF with all matches
    :param date: Giving date by user - asking for games before this date
    :param team: team_id
    :return: last 5 matches for the team (no matter if home or away)
    """
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:5, :]
    return last_matches


def get_last_5_matches_score(matches, team_id):
    '''
    :param matches: DF with last 5 matches the team played
    :return: score team earned in the last 5 matches
    '''
    score_in_5_matches = 0
    for index, row in matches.iterrows():
        home_team = row['home_team_api_id']
        away_team = row['away_team_api_id']
        home_team_goals = row['home_team_goal']
        away_team_goals = row['away_team_goal']
        # final_result = row['final_result']
        if home_team_goals == away_team_goals:
            score_in_5_matches += 1
        elif (home_team_goals > away_team_goals and home_team == team_id) or (
                home_team_goals < away_team_goals and away_team == team_id):
            score_in_5_matches += 3
    return score_in_5_matches


def add_last_5_matches_score(cleared_matches):
    """
    This method will add 2 new columns to the cleared_matches DF with the amount of point
    each team earned in their last 5 home or away games
    :param cleared_matches: - DF with all the matches
    :return: -
    """
    last_5_games_score_home = []
    last_5_games_score_away = []

    for index, match in cleared_matches.iterrows():
        home_team = match['home_team_api_id']
        away_team = match['away_team_api_id']
        match_date = match['date']

        home_team_last_5_games = get_last_5_matches(cleared_matches, match_date, home_team)
        away_team_last_5_games = get_last_5_matches(cleared_matches, match_date, away_team)

        home_team_last_5_score = get_last_5_matches_score(home_team_last_5_games, home_team)
        away_team_last_5_score = get_last_5_matches_score(away_team_last_5_games, away_team)

        last_5_games_score_home.append(home_team_last_5_score)
        last_5_games_score_away.append(away_team_last_5_score)

    cleared_matches['h_5_games_pts'] = last_5_games_score_home
    cleared_matches['a_5_games_pts'] = last_5_games_score_away


def get_teams_rating(row):
    '''
    :param row: row from dataframe
    :return:  new row in dataframe with new column
    1. home_top_player
    2. away_top_player
    3. home_team_mean_rating
    4. away_team_mean_rating
    '''
    '''
    Algorithm:
    1. Extract players id and divide it into home id's and away id's
    2. for each player find the closest data to extract current ratings
    3. based on each player ratings find team average and number of top players.
    '''

    match_date = row['date']
    home_player_1_index = row.keys().tolist().index('home_player_1')
    away_player_1_index = row.keys().tolist().index('away_player_1')
    home_players_api_id = row[
                          home_player_1_index:home_player_1_index + 11]  # list of home players ids who played the game
    away_players_api_id = row[
                          away_player_1_index:away_player_1_index + 11]  # list of away players ids who played the game
    home_players_ratings = []
    away_players_ratings = []

    for id in home_players_api_id:  # loop over all players from home team
        if np.isnan(id):
            home_players_ratings.append(0)
            continue
        id_df = cleared_players[
            cleared_players['player_api_id'] == id]  # id_df = 'player_api_id', 'date', 'overall_rating', 'potential'
        if id_df.empty:
            home_players_ratings.append(0)
            continue
        dates = id_df['date']  # game date

        closest_date = get_closest_date_to_match(match_date, list(dates))

        if closest_date == 0:
            home_players_ratings.append(0)
            continue

        id_df = id_df[id_df['date'] == closest_date]
        ratings = id_df[['overall_rating', 'potential']]
        mean_rating = float((ratings['overall_rating'] + ratings['potential']) / 2)
        home_players_ratings.append(mean_rating)

    for id in away_players_api_id:
        if np.isnan(id):
            home_players_ratings.append(0)
            continue
        id_df = cleared_players[cleared_players['player_api_id'] == id]
        if id_df.empty:
            home_players_ratings.append(0)
            continue
        dates = id_df['date']
        closest_date = get_closest_date_to_match(match_date, list(dates))
        if closest_date == 0:
            away_players_ratings.append(0)
            continue
        ratings = id_df[id_df['date'] == closest_date][['overall_rating', 'potential']]
        mean_rating = float((ratings['overall_rating'] + ratings['potential']) / 2)
        away_players_ratings.append(mean_rating)

    top_rating_level = 80

    row['home_top_players'] = len([x for x in home_players_ratings if x > top_rating_level])
    row['away_top_players'] = len([x for x in away_players_ratings if x > top_rating_level])
    row['home_team_mean_rating'] = np.average(
        [x if x > 0 else np.average([i for i in home_players_ratings if i > 0]) for x in home_players_ratings])
    row['away_team_mean_rating'] = np.average(
        [x if x > 0 else np.average([i for i in away_players_ratings if i > 0]) for x in away_players_ratings])
    return row


def bet_avg_calc_H():
    """
    :param row:
    :return:
    1. home_top_player
    2. away_top_player
    3. home_team_mean_rating
    4. away_team_mean_rating
    """
    bet_sitesH = ["B365H", "BWH", "IWH", "LBH", "WHH", "VCH"]
    bet_sitesH_table = cleared_matches[bet_sitesH]

    cleared_matches['AVG_bet_H'] = np.mean(bet_sitesH_table, axis=1)


def bet_avg_calc_A_D():
    """
        :param row: row from dataframe
    :return:  new row in dataframe with new column
    1. home_top_player
    2. away_top_player
    3. home_team_mean_rating
    4. away_team_mean_rating
    """
    awayDrawAvgTable = pd.DataFrame()
    bet_sitesA = ["B365A", "BWA", "IWA", "LBA", "WHA", "VCA"]
    bet_sitesD = ["B365D", "BWD", "IWD", "LBD", "WHD", "VCD"]
    bet_sitesA_table = cleared_matches[bet_sitesA]
    bet_sitesD_table = cleared_matches[bet_sitesD]

    for col in bet_sitesA:
        colPrefix = col[:-1]
        avgOfCols = ((bet_sitesA_table[colPrefix + 'A'] + bet_sitesD_table[colPrefix + 'D']) / 4).values
        awayDrawAvgTable[colPrefix] = avgOfCols
    tmp = np.mean(awayDrawAvgTable, axis=1)
    print(len(tmp))
    cleared_matches['AVG_bet_A_D'] = tmp


def get_final_results(cleared_matches):
    """EXTRACT FINAL GAMES RESULTS """
    """ NEW COLUMN WILL BE ADDED AFTER THIS PART - FINAL RESULT """
    """  1-Home team won 0-Draw or win for the away team """
    amount_of_games = cleared_matches.shape[0]
    games_results = list(range(amount_of_games))  # will hold all games resultts
    index = 0
    for row, col in cleared_matches.iterrows():
        if (cleared_matches.loc[row].home_team_goal > cleared_matches.loc[row].away_team_goal):
            games_results[index] = 1
        else:
            games_results[index] = 0
        index += 1
    return games_results


def get_final_results_with_draw(cleared_matches):
    """EXTRACT FINAL GAMES RESULTS """
    """ NEW COLUMN WILL BE ADDED AFTER THIS PART - FINAL RESULT """
    """  0-Draw 1-Home team won 2-Away team won """
    amount_of_games = cleared_matches.shape[0]
    games_results = list(range(amount_of_games))  # will hold all games resultts
    index = 0
    for row, col in cleared_matches.iterrows():
        if (cleared_matches.loc[row].home_team_goal > cleared_matches.loc[row].away_team_goal):
            games_results[index] = 1
        elif (cleared_matches.loc[row].home_team_goal < cleared_matches.loc[row].away_team_goal):
            games_results[index] = 2
        elif (cleared_matches.loc[row].home_team_goal == cleared_matches.loc[row].away_team_goal):
            games_results[index] = 0
        index += 1
    cleared_matches['final_result_with_draw'] = games_results


def get_team_attributes(cleared_matches, team_attributes):
    """
    This method will add the buildUpPlaySpeed, defencePressure, defenceAggression, chanceCreationPassing and
    chanceCreationShooting for the home and away team to the cleared_matches DF
    :param cleared_matches: DF with all matches
    :param team_attributes: DF with all team_attributes
    :return:
    """
    buildUpPlaySpeedHome = []
    defencePressureHome = []
    defenceAggressionHome = []
    chanceCreationPassingHome = []
    chanceCreationShootingHome = []

    buildUpPlaySpeedAway = []
    defencePressureAway = []
    defenceAggressionAway = []
    chanceCreationPassingAway = []
    chanceCreationShootingAway = []

    for index, match in cleared_matches.iterrows():
        match_date = match['date']
        home_team_id = match['home_team_api_id']
        away_team_id = match['away_team_api_id']
        home_df = team_attributes[team_attributes['team_api_id'] == home_team_id]
        home_dates = home_df['date']
        home_closest_date = get_closest_date_to_match(match_date, list(home_dates))
        away_df = team_attributes[team_attributes['team_api_id'] == away_team_id]
        away_dates = away_df['date']
        away_closest_date = get_closest_date_to_match(match_date, list(away_dates))

        home_df = home_df[home_df['date'] == home_closest_date]
        away_df = away_df[away_df['date'] == away_closest_date]

        # Home features
        if home_df.size != 0:
            homeBUPS = home_df['buildUpPlaySpeed'].values[0]
            buildUpPlaySpeedHome.append(homeBUPS)
            homeDP = home_df['defencePressure'].values[0]
            defencePressureHome.append(homeDP)
            HomeDA = home_df['defenceAggression'].values[0]
            defenceAggressionHome.append(HomeDA)
            HomeCCP = home_df['chanceCreationPassing'].values[0]
            chanceCreationPassingHome.append(HomeCCP)
            HomeCCS = home_df['chanceCreationShooting'].values[0]
            chanceCreationShootingHome.append(HomeCCS)

        else:
            buildUpPlaySpeedHome.append('')
            defencePressureHome.append('')
            defenceAggressionHome.append('')
            chanceCreationPassingHome.append('')
            chanceCreationShootingHome.append('')

        # Away features
        if away_df.size != 0:
            awayBDPS = away_df['buildUpPlaySpeed'].values[0]
            buildUpPlaySpeedAway.append(awayBDPS)
            awayDP = away_df['defencePressure'].values[0]
            defencePressureAway.append(awayDP)
            awayDA = away_df['defenceAggression'].values[0]
            defenceAggressionAway.append(awayDA)
            awayCCP = away_df['chanceCreationPassing'].values[0]
            chanceCreationPassingAway.append(awayCCP)
            awayCCS = away_df['chanceCreationShooting'].values[0]
            chanceCreationShootingAway.append(awayCCS)
        else:
            buildUpPlaySpeedAway.append('')
            defencePressureAway.append('')
            defenceAggressionAway.append('')
            chanceCreationPassingAway.append('')
            chanceCreationShootingAway.append('')

    cleared_matches['buildUpPlaySpeedHome'] = buildUpPlaySpeedHome
    cleared_matches['buildUpPlaySpeedAway'] = buildUpPlaySpeedAway
    cleared_matches['defencePressureHome'] = defencePressureHome
    cleared_matches['defencePressureAway'] = defencePressureAway
    cleared_matches['defenceAggressionHome'] = defenceAggressionHome
    cleared_matches['defenceAggressionAway'] = defenceAggressionAway
    cleared_matches['chanceCreationPassingHome'] = chanceCreationPassingHome
    cleared_matches['chanceCreationPassingAway'] = chanceCreationPassingAway
    cleared_matches['chanceCreationShootingHome'] = chanceCreationShootingHome
    cleared_matches['chanceCreationShootingAway'] = chanceCreationShootingAway


"******************************** END OF SCRIPT FUNCTIONS ********************************"
"******************** SCRIPT TO BUILD DATAFRAME WITH SPECIAL FEATURES ********************"

path = "D:/School/20-21/Semester F/Project Preparation Workshop in Information Systems Engineering/Project 2021/HW3/database.sqlite"

database = path
con = sqlite3.connect(database)
cur = con.cursor()

"""TABLES"""
players_stats = pd.read_sql("SELECT * FROM Player_Attributes;", con)
match_stats = pd.read_sql(
    'SELECT * FROM Match WHERE season in ("2015/2016","2014/2015","2013/2014","2012/2013","2011/2012");', con)

team_attributes = pd.read_sql("SELECT * FROM Team_Attributes;", con)
team_needed_featuers = ["team_api_id", "date", "buildUpPlaySpeed", "defencePressure", "defenceAggression",
                        "chanceCreationPassing", "chanceCreationShooting"]
team_df = team_attributes[team_needed_featuers]
team_df['date'] = pd.to_datetime(team_df['date'])

# match_stats = pd.read_sql('SELECT * FROM Match WHERE season in ("2015/2016","2014/2015");',con)
match_features = ["id", "date", "season", "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal",
                  "home_player_1", "home_player_2", "home_player_3", "home_player_4", "home_player_5", "home_player_6",
                  "home_player_7", "home_player_8", "home_player_9", "home_player_10", "home_player_11",
                  "away_player_1", "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
                  "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11",
                  "B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA", "LBH", "LBD", "LBA", "WHH",
                  "WHD", "WHA", "VCH", "VCD", "VCA"]
players_features = ['player_api_id', 'date', 'overall_rating', 'potential']

"""CLEAR TABLES"""
# Table contains data from seasons 2011/2012 - 2015/2016 (2015/2016 will be splited later on)
cleared_matches = match_stats[match_features].replace(to_replace='None', value=np.nan).dropna()
print("Starting to process...: \n", cleared_matches)

""" Adding final_resul to the table """
games_results = get_final_results(cleared_matches)
cleared_matches['final_result'] = games_results
get_final_results_with_draw(cleared_matches)  # add results with draw option
print("Added games results ...: \n\n", cleared_matches)


#
# """ ADDING buildUpPlaySpeed defencePressure defenceAggression chanceCreationPassing chanceCreationShooting TO HOME AND AWAY SIDE """
# # get_team_attributes(cleared_matches, team_df)
# # cleared_matches = cleared_matches.replace(to_replace='', value=np.nan).dropna()
# # print("Added Team Attributes ...: \n\n", cleared_matches)
#
#

""" ADDING THE AMOUNT OF POINTS EACH TEAM GOT IN THE PREVIOUSLY 5 GAMES TO THE CURRENT GAME"""
add_last_5_matches_score(cleared_matches)
print("Added 5 games points results ...: \n\n", cleared_matches)

x_games = 10
add_last_x_matches_score(cleared_matches, x_games)
print("Added" + str(x_games) + "games points results ...: \n\n", cleared_matches)

""" ADDING THE BET'S AVERAGE ODDS TO HOME, AWAY AND DRAW RESULTS """
bet_avg_calc_H()
bet_avg_calc_A_D()
cleared_matches = cleared_matches.dropna(how='any')
print("Added bets ...: \n\n", cleared_matches)

""" FIND OUT HOW MANY TOP PLAYERS PLAYED FOR THE TEAM """
""" TOP PLAYER IS A PLAYER WITH AVG OVERALL HIGHER THEN 80 """
""" AVG OVERALL = (overall_rating + potential) / 2 """
""" BY THE END OF THIS SECTION THE TABLE WILL HAVE 4 NEW COLUMNS """
""" TOP_PLAYERS_HOME_TEAM AND TOP_PLAYERS_AWAY_TEAM """
""" AVG OF THE PLAYERS RATING FOR HOME AND AWAY TEAM """
cleared_players = players_stats[
    players_features]  # New table holds players 'player_api_id', 'date', 'overall_rating', 'potential'
cleared_players['date'] = pd.to_datetime(cleared_players['date'])
cleared_players.dropna(inplace=True)
cleared_matches = cleared_matches.apply(get_teams_rating, axis=1)
print("Added AVG OVERALL and team Top players ...: \n\n", cleared_matches)

""" Split to test and train DFs """
print("Splitting to train and test... \n")

needed_features = ['home_team_api_id', 'away_team_api_id', 'home_team_mean_rating', 'away_team_mean_rating',
                   'home_top_players','away_top_players','AVG_bet_H','AVG_bet_A_D','h_5_games_pts',
                   'a_5_games_pts','h_only_10_games_pts','a_only_10_games_pts',
                   'final_result']

train_df = pd.DataFrame(columns=needed_features)
test_df = pd.DataFrame(columns=needed_features)

for index, row in cleared_matches.iterrows():
    season = row['season']
    to_add = row[needed_features]
    if season == '2015/2016':
        test_df = test_df.append(to_add)
        # add to test DF
    else:
        # add to trainning DF
        train_df = train_df.append(to_add)

print("Done split: \n")
print("Train DF: \n\n\n", train_df)
print("Test DF: \n\n\n", test_df)

print("Export to CSV...\n")
test_df.to_csv('test.csv')
train_df.to_csv('train.csv')

print("Export to CSV is OVER\n")

print("Export full features...\n")

finalDF = cleared_matches[needed_features]
finalDF.to_csv('full_features.csv')

print("Export full features is OVER\n")
