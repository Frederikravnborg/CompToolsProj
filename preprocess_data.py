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

# 4. Encode 'primary_position' using LabelEncoder for validation purposes
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
        return 'Off'  # Map 'Other' to 'Off'

df['player_positions_simple'] = df['primary_position'].apply(map_position_simple)

# 6. Define all possible categories for one-hot encoding

# Simplified position categories (only four)
simplified_categories = ['GK', 'Def', 'Mid', 'Off']
df['player_positions_simple'] = pd.Categorical(df['player_positions_simple'], categories=simplified_categories)

# Original position categories (list all possible positions)
original_categories = [
    'GK', 'CB', 'LB', 'RB', 'LWB', 'RWB',
    'CDM', 'CM', 'CAM', 'LM', 'RM',
    'RW', 'LW', 'ST', 'CF', 'Unknown'
]
df['primary_position'] = pd.Categorical(df['primary_position'], categories=original_categories)

# 7. Identify all relevant columns for clustering and validation

# Features for clustering
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

# Columns to retain for validation and inspection
validation_columns = [
    'short_name', 'sofifa_id', 'player_positions_enc', 'primary_position', 'player_positions_simple'
]

# 8. Handle Missing Values by Imputation

# Define all relevant columns to handle missing values (both clustering and validation)
all_relevant_columns = selected_features + validation_columns

# Separate the data into features and validation/identifier columns
features_df = df[selected_features]
validation_df = df[validation_columns]

# Define feature types
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

# Initialize imputers
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute numerical features
features_numerical = features_df[numerical_features]
features_numerical_imputed = pd.DataFrame(numerical_imputer.fit_transform(features_numerical), columns=numerical_features)

# Impute categorical features
features_categorical = features_df[categorical_features]
features_categorical_imputed = pd.DataFrame(categorical_imputer.fit_transform(features_categorical), columns=categorical_features)

# Correct any invalid 'body_type' entries
valid_body_types = ['Lean', 'Normal', 'Stocky']
features_categorical_imputed['body_type'] = np.where(
    features_categorical_imputed['body_type'].isin(valid_body_types),
    features_categorical_imputed['body_type'],
    'Normal'  # Default to 'Normal' if invalid
)

# Update the main DataFrame with imputed features
df_cleaned = pd.concat([features_numerical_imputed, features_categorical_imputed, validation_df.reset_index(drop=True)], axis=1)

print(f"Number of rows after handling missing values: {df_cleaned.shape[0]}")

# 9. One-hot encode simplified positions for validation
onehot_simple = pd.get_dummies(df_cleaned['player_positions_simple'], prefix='player_positions_simple_enc')

# One-hot encode original primary positions for validation
onehot_orig = pd.get_dummies(df_cleaned['primary_position'], prefix='player_positions_orig_enc')

# 10. Verify that all expected one-hot encoded columns are present
# For simplified positions
for category in simplified_categories:
    col_name = f'player_positions_simple_enc_{category}'
    if col_name not in onehot_simple.columns:
        onehot_simple[col_name] = 0
        print(f"Added missing simplified position column: {col_name}")

# For original positions
for category in original_categories:
    col_name = f'player_positions_orig_enc_{category}'
    if col_name not in onehot_orig.columns:
        onehot_orig[col_name] = 0
        print(f"Added missing original position column: {col_name}")

# 11. Encode Simplified Positions as Integers (0 to 3)
# Define mapping for simplified positions
position_mapping = {'GK': 0, 'Def': 1, 'Mid': 2, 'Off': 3}
df_cleaned['player_positions_simple_enc'] = df_cleaned['player_positions_simple'].map(position_mapping)

# 12. Define feature types (Already defined above)

# 13. Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# 14. Apply preprocessing to clustering features
preprocessed_features = preprocessor.fit_transform(df_cleaned[selected_features])

# 15. Convert preprocessed features to DataFrame
onehot_features_cat = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_features = numerical_features + list(onehot_features_cat)
preprocessed_df = pd.DataFrame(preprocessed_features, columns=all_features)

# 16. Ensure no remaining NaNs in preprocessed features
if preprocessed_df.isnull().values.any():
    print("Warning: There are still NaN values in the preprocessed features.")
else:
    print("No NaN values found in the preprocessed features.")

# 17. Combine validation columns with preprocessed features
# Position encodings and identifiers are for validation only and should not influence clustering
validation_df = pd.concat([
    df_cleaned[['short_name', 'sofifa_id', 'player_positions_enc', 'primary_position', 'player_positions_simple_enc']].reset_index(drop=True),
    onehot_simple.reset_index(drop=True),
    onehot_orig.reset_index(drop=True)
], axis=1)

# Reorder columns to have identifiers and position encodings first
validation_columns_order = [
    'short_name', 'sofifa_id', 'player_positions_enc', 'primary_position', 'player_positions_simple_enc'
] + [f'player_positions_simple_enc_{cat}' for cat in simplified_categories] + \
    [f'player_positions_orig_enc_{cat}' for cat in original_categories]

validation_df = validation_df[validation_columns_order]

# 18. Concatenate the preprocessed features with the validation columns
final_df = pd.concat([
    validation_df,
    preprocessed_df
], axis=1)

# 19. Verify row counts
if final_df.shape[0] != df_cleaned.shape[0]:
    print(f"Warning: Row count mismatch between preprocessed features and validation data. Preprocessed features have {preprocessed_df.shape[0]} rows, validation data has {validation_df.shape[0]} rows.")
else:
    print("Row counts are consistent between preprocessed features and validation data.")

# Include a column with the simple position titles
position_mapping = {
    0: 'GK',
    1: 'Def',
    2: 'Mid',
    3: 'Off'
}
# Creating a new column with mapped values
final_df.insert(4, 'primary_position_simple', final_df['player_positions_simple_enc'].map(position_mapping))




# 20. Save the final preprocessed data
final_df.to_csv('players_preprocessed.csv', index=False)

print("Preprocessing completed successfully!")
print("'short_name' and 'sofifa_id' are retained for inspection.")
print("'player_positions_enc' and 'primary_position' are retained for validation.")
print("'player_positions_simple_enc' is created for simplified position encoding (0: GK, 1: Def, 2: Mid, 3: Off).")
print("One-hot encoded simplified and original positions are retained for validation.")
print("Clustering features are separate and do not include any position-related encodings or identifier columns.")
