import pandas as pd
import os


def get_stats_previous_matchs(team_id, match_date, n_match=5):
    #root_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join('..', 'raw_data', 'teamstats.csv')
    stat = pd.read_csv(csv_path).copy()
    #Convert to date_time and sort
    stat['date'] = pd.to_datetime(stat['date'])
    stat = stat.sort_values(by='date')
    #Filter by team id and previous matchs
    team_stat = stat[stat['teamID'] == team_id]
    stat_5_match = team_stat[team_stat['date'] < match_date].tail(
        n_match).sum()
    stat_5_match.drop(['gameID', 'season'], inplace=True)
    return pd.DataFrame(stat_5_match.values.reshape(1, -1),
                        columns=stat_5_match.index)


def get_basic_data():

    games = pd.read_csv('../raw_data/games.csv').copy()
    #Drop bet columns
    games = games[[
        'gameID', 'leagueID', 'season', 'date', 'homeTeamID', 'awayTeamID',
        'homeGoals', 'awayGoals'
    ]]
    #convert to datetime
    games['date'] = pd.to_datetime(games['date'])
    #Compute Target
    games['result'] = games['homeGoals'] - games['awayGoals']
    games['result'] = games['result'].apply(lambda x: 'D' if x == 0 else 'W'
                                            if x > 0 else 'L')
    games.drop(columns=['homeGoals', 'awayGoals'], inplace=True)
    #games_2015 = games[games['season'] == 2015]
    #games_2015.reset_index(inplace=True)

    df_away = pd.DataFrame()
    df_home = pd.DataFrame()
    for index, row in games.iterrows():
        #Get stats for home team
        team_id = row['homeTeamID']
        match_date = row['date']
        home_stats = get_stats_previous_matchs(team_id, match_date)
        #Get stats for away team
        team_id = row['awayTeamID']
        away_stats = get_stats_previous_matchs(team_id, match_date)
        #Append new row in dataframe
        df_home = pd.concat([df_home, home_stats])
        df_away = pd.concat([df_away, away_stats])

    #Rename columns with prefix Away or Home
    df_away = df_away.rename(
        columns={old_name: 'Away' + old_name
                 for old_name in df_away.columns})
    df_home = df_home.rename(
        columns={old_name: 'Home' + old_name
                 for old_name in df_home.columns})
    #Construct final df
    df = pd.concat([df_home, df_away], axis=1)
    games.reset_index(inplace=True)
    df.reset_index(inplace=True)
    #Add target and drop useless columns
    df['target'] = games['result']
    df.drop(columns=['index', 'HometeamID', 'AwayteamID'], inplace=True)
    #Add gameID
    df['gameID'] = games['gameID']

    return df
