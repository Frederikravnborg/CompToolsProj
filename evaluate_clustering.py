# Import necessary libraries
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset with cluster labels
df = pd.read_csv('players_with_clusters.csv')

# 2. Inspect the 'player_positions' column
print("Unique Player Positions:")
print(df['player_positions'].unique())

# 3. Handle multiple positions per player (if applicable)
# Assuming 'player_positions' may contain multiple positions separated by commas
# We'll take the first position as the primary position for simplicity
df['primary_position'] = df['player_positions'].apply(lambda x: x.split(',')[0].strip())

# 4. Create a Crosstab to Compare Clusters with Actual Positions
crosstab = pd.crosstab(df['Cluster'], df['primary_position'])
print("\nCrosstab of Clusters vs. Primary Player Positions:")
print(crosstab)

# 5. Visualize the Crosstab with a Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
plt.title('Heatmap of Clusters vs. Primary Player Positions')
plt.xlabel('Primary Player Position')
plt.ylabel('Cluster')
plt.show()

# 6. Calculate External Validation Metrics

# Ensure that both 'Cluster' and 'primary_position' are numerical or categorical labels
# For metrics like ARI and NMI, labels need to be encoded numerically
# We'll use label encoding for 'primary_position'

le = LabelEncoder()
df['position_encoded'] = le.fit_transform(df['primary_position'])

# Calculate Adjusted Rand Index (ARI)
ari = adjusted_rand_score(df['position_encoded'], df['Cluster'])
print(f"\nAdjusted Rand Index (ARI) between clusters and actual positions: {ari:.4f}")

# Calculate Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(df['position_encoded'], df['Cluster'])
print(f"Normalized Mutual Information (NMI) between clusters and actual positions: {nmi:.4f}")

# 7. Optional: Analyze Cluster Purity
# Assign the most common actual position to each cluster
cluster_to_position = crosstab.idxmax(axis=1)
df['cluster_primary_position'] = df['Cluster'].map(cluster_to_position)

# Calculate the percentage of players in each cluster that match the assigned position
cluster_purity = (df['primary_position'] == df['cluster_primary_position']).groupby(df['Cluster']).mean() * 100
print("\nCluster Purity (% of players matching the most common position in the cluster):")
print(cluster_purity)

# 8. Visualize Cluster Purity
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_purity.index, y=cluster_purity.values, palette='viridis')
plt.title('Cluster Purity by Assigned Position')
plt.xlabel('Cluster')
plt.ylabel('Purity (%)')
plt.ylim(0, 100)
plt.show()
