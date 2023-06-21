##################### PREWORK #######################

import numpy as np
import pandas as pd
import datetime 
import time
from github import Github
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def update_repository():
    # GitHub credentials
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

        # Read the updated file contents
    with open('code_model_1.py', 'r') as file:
        updated_contents = file.read()

    file = repo.get_contents('code_model_1.py', ref=branch_ref)

    # Update the file
    repo.update_file(
        file.path,
        "Commiteado con python",
        updated_contents,
        file.sha
    )

    print("Repository updated successfully!")
    
# Specify the path to the JSON file
git_credential = "git_credential.json"

# Load the JSON object from the file
with open(git_credential, "r") as file:
    credentials = json.load(file)

access_token = credentials["access_token"]

update_repository()

##################### 2-LIMPIO DATA DE PARTIDAS #######################

game_data=pd.read_csv('partidas_analized_historico.csv')

game_data=game_data.drop_duplicates()

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

winners['result']=np.where(winners['team']=='red',1,0)

game_data_summary=pd.merge(game_data_summary,winners[['game','result']],how='left',on='game')


############################ MODELO BY POSITION


feature_importances_df=pd.DataFrame(columns=['victories_overall', 
       'winrate_overall', 'winrate_last_5_games',
       'winrate_last_10_games', 'winrate_last_20_games',
       'winrate_last_50_games', 'victories_position', 'winrate_position',
       'pickrate_position', 'victories_champ', 'winrate_champ',
       'pickrate_champ'])

for position in tqdm(range(1,6,1)):
    
    regex_red=f'red_{position}_|^result$'
    
    red = game_data_summary.filter(regex=regex_red)
    
    red['side_red']=1
    
    red_column=f'red_{position}_'
    red.columns=red.columns.str.replace(red_column, '')
    
    regex_blue=f'blue_{position}_|^result$'
    blue = game_data_summary.filter(regex=regex_blue)
    
    blue['side_red']=0
    
    blue_column=f'blue_{position}_'
    blue.columns=blue.columns.str.replace(blue_column, '')
    
    
    df=pd.concat([red,blue])
    
    
    ############################### MODEL
        
    df = df.drop('side_red', axis=1)  # Remove the target column from the features
    
    X = df.drop('result', axis=1)  # Remove the target column from the features
    y = df['result']               # Select the target column
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Normalize the features
    normalizer = MinMaxScaler()
    X_train_norm = normalizer.fit_transform(X_train_std)
    X_test_norm = normalizer.transform(X_test_std)
    
    
    models = [
        ("Logistic Regression", LogisticRegression()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("Gradient Boosting", GradientBoostingClassifier())
    ]
    
    scores=[]
    importance=[]
    modelos=[]
    for name, classifier in models:
        modelos=modelos+[name]
        # Train the classifier
        classifier.fit(X_train_norm, y_train)
        
        # Make predictions on the test set
        y_pred = classifier.predict(X_test_norm)
        
        # Evaluate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        scores=scores+[accuracy]
        try:
            importance_case=classifier.coef_
        except Exception:
            importance_case = classifier.feature_importances_
        importance=importance+[importance_case]
    
    feature_importances_df_position=pd.DataFrame(columns=X_train.columns)
    
    feature_importances_df_position=pd.concat([feature_importances_df_position,pd.DataFrame(importance[0].tolist(),columns=X_train.columns)])
    
    feature_importances_df_position=pd.concat([feature_importances_df_position,pd.DataFrame([importance[1].tolist()],columns=X_train.columns)])
    
    feature_importances_df_position=pd.concat([feature_importances_df_position,pd.DataFrame([importance[2].tolist()],columns=X_train.columns)])
    
    feature_importances_df_position=pd.concat([feature_importances_df_position,pd.DataFrame([importance[3].tolist()],columns=X_train.columns)])
    
    feature_importances_df_position.index=modelos
    
    feature_importances_df_position['score']=scores
    
    feature_importances_df_position['position']=position

    feature_importances_df=pd.concat([feature_importances_df,feature_importances_df_position])