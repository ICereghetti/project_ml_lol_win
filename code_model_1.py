##################### PREWORK #######################

import requests
import numpy as np
import pandas as pd
from time import sleep
import datetime 
from datetime import timedelta
import time
import os
from tqdm import tqdm
import cProfile
from itertools import chain
from ast import literal_eval  # For safely evaluating string literals as lists


##################### 2-LIMPIO DATA DE PARTIDAS #######################

game_data=pd.read_csv('partidas_analized_historico.csv')

game_data['game_date']=np.NaN

today = datetime.date.today()

game_data=game_data.reset_index(drop=True)

#Arreglo fecha
for i in range(len(game_data)):
    epoch_time=game_data['gameCreation'][i]
    game_data['game_date'][i]=time.strftime('%Y-%m-%d', time.localtime(epoch_time/1000))

#Arreglo wins a int
game_data.win = game_data.win.replace({True: 1, False: 0})

# Rank to last games

game_data['game_rank'] = game_data.sort_values(['gameCreation']).groupby('puuid').cumcount()+1

################################# I GET INFORMATION FOR EACH PLAYER


##### DF 1 : WINRATES OVERALL AND IN LAST GAMES

player_resultados_base=game_data.groupby('puuid').agg({
    'win': ['count', 'mean',
            lambda x: x[game_data['game_rank'] <= 5].mean(),
            lambda x: x[game_data['game_rank'] <= 10].mean(),
            lambda x: x[game_data['game_rank'] <= 20].mean(),
            lambda x: x[game_data['game_rank'] <= 50].mean()]
})

player_resultados_base.columns = ['victories_overall', 'winrate_overall',
                                  'winrate_last_5_games', 'winrate_last_10_games',
                                  'winrate_last_20_games', 'winrate_last_50_games']


player_resultados_position=game_data.groupby(['puuid','individualPosition','role']).agg({
    'win': ['count', 'mean']
})

player_resultados_position.columns = ['victories_position', 'winrate_position']

player_resultados_position['pickrate_position'] = player_resultados_position.groupby(['puuid'])['victories_position'].transform(lambda x: x / x.sum() * 100)

##### DF 1 : WINRATES BY CHAMPION

player_resultados_champ=game_data.groupby(['puuid','championName']).agg({
    'win': ['count', 'mean']
})


player_resultados_champ.columns = ['victories_champ', 'winrate_champ']

player_resultados_champ['pickrate_champ'] = player_resultados_champ.groupby(['puuid'])['victories_champ'].transform(lambda x: x / x.sum() * 100)


player_resultados_champ=game_data.groupby(['puuid','championName']).agg({
    'win': ['count', 'mean']
})

player_resultados_champ.columns = ['victories_champ', 'winrate_champ']

player_resultados_champ['pickrate_champ'] = player_resultados_champ.groupby(['puuid'])['victories_champ'].transform(lambda x: x / x.sum() * 100)

#### number

#### Player number

game_data['player_order'] = game_data.groupby('game').cumcount()+1

game_data['team'] = np.where(game_data['player_order'] <= 5, 'red', 'blue')

game_data['player_team_order'] = game_data.groupby(['game','team']).cumcount()+1

####Completo con data las partidas

game_data_complete=pd.merge(game_data,player_resultados_base,how='left',on='puuid')

game_data_complete=pd.merge(game_data_complete,player_resultados_position,
                            how='left',on=['puuid','individualPosition','role'])

game_data_complete=pd.merge(game_data_complete,player_resultados_champ,
                            how='left',on=['puuid','championName'])

#### Re-armo ultimas partidas

game_data_summary=pd.pivot(game_data_complete, index=['game','gameCreation','gameVersion','game_date'], columns=['team','player_team_order'], 
                           values=['victories_overall', 'winrate_overall',
                                   'winrate_last_5_games', 'winrate_last_10_games',
                                   'winrate_last_20_games', 'winrate_last_50_games', 
                                   'victories_position','winrate_position', 
                                   'pickrate_position', 'victories_champ',
                                   'winrate_champ', 'pickrate_champ'])


# Define the desired column order
game_data_summary.columns = game_data_summary.columns.reorder_levels([1, 2, 0])

game_data_summary.columns = ['_'.join([str(col) for col in cols]) for cols in game_data_summary.columns]

winners=game_data[game_data['win']==1][['game','team']].drop_duplicates()

winners['result']=np.where(winners['team']=='team_red',1,0)

game_data_summary=pd.merge(game_data_summary,winners[['game','result']],how='left',on='game')


############################ MODELO BY POSITION

from github import Github

def update_repository():
    # GitHub credentials
    access_token = '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'  # Replace with your access token
    repo_owner = 'ICereghetti'  # Replace with your username
    repo_name = 'project_ml_lol_win'  # Replace with your repository name

    # Create a GitHub instance using your access token
    g = Github(access_token)

    # Get the repository
    repo = g.get_user(repo_owner).get_repo(repo_name)

    # Get the default branch
    default_branch = repo.default_branch

    # Get the branch reference
    branch_ref = repo.get_branch(default_branch).commit.sha

    # Create a new file commit
    commit = repo.create_git_commit(
        "Update from script",
        "YOUR_FILE_NAME",
        "YOUR_COMMIT_MESSAGE",
        branch_ref,
        "YOUR_FILE_CONTENTS"
    )

    # Update the branch reference
    ref = repo.get_git_ref(f'heads/{default_branch}')
    ref.edit(commit.sha)

    print("Repository updated successfully!")

# Your existing code goes here

# Call the update_repository() function at the end of your code's execution
update_repository()
    
update_repository()


