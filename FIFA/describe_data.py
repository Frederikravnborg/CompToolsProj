import pandas as pd
import numpy as np

desired_columns = ['age','height_cm','weight_kg','nationality','club','overall',
                   'value_eur','wage_eur','player_positions','preferred_foot','work_rate','weak_foot',
                   'pace','shooting','passing','dribbling','defending','physic']
df = pd.read_csv("players_20.csv", usecols=desired_columns)
df_headers = df.columns
df_summary = df.describe()
df_covariance_matrix = df.cov()


columns = ['nationality','club','player_positions','work_rate']
lll = df.drop(columns)
print(lll)