import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the dataset
df = pd.read_csv('players_20.csv')

# 2. List of columns to remove
columns_to_remove = [
    'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw',
    'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm',
    'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb',
    'cb', 'rcb', 'rb'
]

# Drop the specified columns
df.drop(columns=columns_to_remove, inplace=True)

# 3. Extract and process 'player_positions'

# Function to extract the first position
def extract_primary_position(positions):
    if pd.isnull(positions):
        return 'Unknown'
    return positions.split(',')[0].strip()

# Apply the function to create 'primary_position'
df['primary_position'] = df['player_positions'].apply(extract_primary_position)

# 4. Encode 'primary_position' using LabelEncoder for validation
le = LabelEncoder()
df['player_positions_enc'] = le.fit_transform(df['primary_position'])

# 5. Map primary positions to simplified categories
def map_position_simple(position):
    if position == 'GK':
        return 'GK'
    elif position in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'Def'
    elif position in ['CDM', 'CM', 'CAM', 'LM', 'RM']:
        return 'Mid'
    elif position in ['RW', 'LW', 'ST', 'CF']:
        return 'Off'
    else:
        return 'Other'  # For any positions not specified

df['player_positions_simple'] = df['primary_position'].apply(map_position_simple)

# 6. One-hot encode simplified positions
onehot_simple = pd.get_dummies(df['player_positions_simple'], prefix='player_positions_simple_enc')

# 7. One-hot encode original primary positions
onehot_orig = pd.get_dummies(df['primary_position'], prefix='player_positions_orig_enc')

# 8. Select relevant features for clustering
selected_features = [
    'age', 'height_cm', 'weight_kg',
    'preferred_foot', 'body_type',
    'pace', 'shooting', 'passing', 'dribbling',
    'defending', 'physic',
    'skill_moves', 'weak_foot',
    'mentality_vision', 'mentality_composure', 'mentality_aggression',
    'mentality_interceptions', 'mentality_positioning',
    'work_rate', 'international_reputation'
]
data = df[selected_features].copy()

# 9. Define feature types
numerical_features = [
    'age', 'height_cm', 'weight_kg',
    'pace', 'shooting', 'passing', 'dribbling',
    'defending', 'physic',
    'skill_moves', 'weak_foot',
    'mentality_vision', 'mentality_composure', 'mentality_aggression',
    'mentality_interceptions', 'mentality_positioning',
    'international_reputation'
]
categorical_features = ['preferred_foot', 'body_type', 'work_rate']

# 10. Correct any invalid 'body_type' entries
valid_body_types = ['Lean', 'Normal', 'Stocky']
data['body_type'] = np.where(data['body_type'].isin(valid_body_types), data['body_type'], 'Normal')

# 11. Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# 12. Apply preprocessing
preprocessed_data = preprocessor.fit_transform(data)

# 13. Convert to DataFrame
onehot_features_cat = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_features = numerical_features + list(onehot_features_cat)
preprocessed_df = pd.DataFrame(preprocessed_data, columns=all_features)

# 14. Handle remaining missing values if any
preprocessed_df.dropna(inplace=True)

# 15. Reset index to align with 'player_positions_enc', 'primary_position', and simplified positions
preprocessed_df.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# 16. Align preprocessed data with the original DataFrame after dropna
# Identify the rows that were kept after dropna
data_non_null = data.dropna().reset_index(drop=True)
df_non_null = df.loc[data_non_null.index].reset_index(drop=True)

# 17. Concatenate 'player_positions_enc', 'primary_position', one-hot encoded simplified and original positions, and preprocessed features
preprocessed_df = pd.concat([
    df_non_null['player_positions_enc'],
    df_non_null['primary_position'],
    onehot_simple.loc[data_non_null.index].reset_index(drop=True),
    onehot_orig.loc[data_non_null.index].reset_index(drop=True),
    preprocessed_df
], axis=1)

# 18. Save preprocessed data with 'player_positions_enc' and 'primary_position' as the first two columns,
# followed by one-hot encoded simplified and original positions, then preprocessed features
preprocessed_df.to_csv('players_preprocessed.csv', index=False)

print("Preprocessing completed successfully!")
