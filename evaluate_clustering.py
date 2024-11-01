# Import necessary libraries
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# Step 0: Define Toggle for Position Encoding
# ============================================

# Toggle to switch between simplified and original position encodings
USE_SIMPLE_POSITIONS = True  # Set to False to use original positions

# ============================================
# Step 1: Load the Clustered Data
# ============================================

# Load the clustered data
clustered_df = pd.read_csv('players_with_clusters.csv')

# ============================================
# Step 2: Select Position Labels Based on Toggle
# ============================================

if USE_SIMPLE_POSITIONS:
    # Check if simplified encoding exists
    if 'player_positions_simple_enc' in clustered_df.columns:
        position_label = clustered_df['player_positions_simple_enc']
        position_name = 'Simplified Position'
        position_column = 'primary_position_simple'
        print("Using simplified position encodings for evaluation.")
    else:
        raise ValueError("Simplified position encodings ('player_positions_simple_encoded') not found in the DataFrame.")
else:
    # Use original position encoding
    if 'player_positions_enc' in clustered_df.columns:
        position_label = clustered_df['player_positions_enc']
        position_name = 'Original Position'
        position_column = 'primary_position'
        print("Using original position encodings for evaluation.")
    else:
        raise ValueError("Original position encodings ('player_positions_enc') not found in the DataFrame.")

# ============================================
# Step 3: Extract Relevant Columns for Validation
# ============================================

cluster_labels = clustered_df['Cluster']

# Extract the actual position labels (string labels) for crosstab and purity calculations
if USE_SIMPLE_POSITIONS:
    simplified_position = clustered_df['primary_position_simple']
    # Ensure that simplified positions are categorical with defined categories
    if not pd.api.types.is_categorical_dtype(simplified_position):
        print("Converting 'primary_position_simple' to categorical.")
        simplified_position = pd.Categorical(simplified_position)
else:
    original_position = clustered_df['primary_position']
    # Ensure that original positions are categorical with defined categories
    if not pd.api.types.is_categorical_dtype(original_position):
        print("Converting 'primary_position' to categorical.")
        original_position = pd.Categorical(original_position)

# ============================================
# Step 4: Create Crosstab and Heatmap
# ============================================

# Create a Crosstab to Compare Clusters with Selected Positions
crosstab = pd.crosstab(cluster_labels, clustered_df[position_column])
print(f"\nCrosstab of Clusters vs. {position_name}:")
print(crosstab)

# Visualize the Crosstab with a Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
plt.title(f'Heatmap of Clusters vs. {position_name}')
plt.xlabel(f'{position_name}')
plt.ylabel('Cluster')
plt.show()

# ============================================
# Step 5: Calculate External Validation Metrics
# ============================================

# Calculate Adjusted Rand Index (ARI)
ari = adjusted_rand_score(position_label, cluster_labels)
print(f"\nAdjusted Rand Index (ARI) between clusters and {position_name.lower()}: {ari:.4f}")

# Calculate Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(position_label, cluster_labels)
print(f"Normalized Mutual Information (NMI) between clusters and {position_name.lower()}: {nmi:.4f}")

# ============================================
# Step 6: Analyze Cluster Purity
# ============================================

# Assign the most common position to each cluster
cluster_to_position = crosstab.idxmax(axis=1)
clustered_df['cluster_assigned_position'] = clustered_df['Cluster'].map(cluster_to_position)

# Calculate the percentage of players in each cluster that match the assigned position
if USE_SIMPLE_POSITIONS:
    # Using simplified positions
    cluster_purity = (clustered_df['player_positions_simple'] == clustered_df['cluster_assigned_position']).groupby(clustered_df['Cluster']).mean() * 100
    purity_title = 'Simplified Position'
else:
    # Using original positions
    cluster_purity = (clustered_df['primary_position'] == clustered_df['cluster_assigned_position']).groupby(clustered_df['Cluster']).mean() * 100
    purity_title = 'Original Position'

print(f"\nCluster Purity (% of players matching the most common {purity_title} in the cluster):")
print(cluster_purity)

# Visualize Cluster Purity
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_purity.index, y=cluster_purity.values, palette='viridis')
plt.title(f'Cluster Purity by Assigned {purity_title}')
plt.xlabel('Cluster')
plt.ylabel('Purity (%)')
plt.ylim(0, 100)
plt.show()
