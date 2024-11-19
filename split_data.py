import pandas as pd
from sklearn.model_selection import train_test_split

# Specify the path to your input CSV file
input_file = '/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms+spam+collection/sms_spam_collection.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Ensure that the 'Tag' column has only 'spam' and 'ham' values
df = df[df['Tag'].isin(['spam', 'ham'])]

# Split the data into training and test sets with stratification to maintain label proportions
train_df, test_df = train_test_split(
    df,
    test_size=0.10,       # 10% for testing
    stratify=df['Tag'],   # Maintain the proportion of 'spam' and 'ham'
    random_state=42       # For reproducibility
)

# Optionally, shuffle the training and test sets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the training set to 'train.csv'
train_df.to_csv('train.csv', index=False)

# Save the test set to 'test.csv'
test_df.to_csv('test.csv', index=False)

print("Data has been successfully split into 'train.csv' and 'test.csv'.")