# Import necessary libraries for clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Determine the number of clusters based on looking at the dendrogram
num_clusters = 4


# Load the preprocessed data
preprocessed_df = pd.read_csv('players_preprocessed.csv')

# Select features for clustering
# Exclude all position-related columns and identifier columns
features = preprocessed_df.drop(
    columns=[
        'short_name',            
        'sofifa_id',             
        'player_positions_enc', 
        'primary_position',
        'primary_position_simple',    
        # Exclude one-hot encoded simplified positions
        'player_positions_simple_enc_Def',
        'player_positions_simple_enc_GK',
        'player_positions_simple_enc_Mid',
        'player_positions_simple_enc_Off',
        # Exclude one-hot encoded original positions
        'player_positions_orig_enc_GK',
        'player_positions_orig_enc_CB',
        'player_positions_orig_enc_LB',
        'player_positions_orig_enc_RB',
        'player_positions_orig_enc_LWB',
        'player_positions_orig_enc_RWB',
        'player_positions_orig_enc_CDM',
        'player_positions_orig_enc_CM',
        'player_positions_orig_enc_CAM',
        'player_positions_orig_enc_LM',
        'player_positions_orig_enc_RM',
        'player_positions_orig_enc_RW',
        'player_positions_orig_enc_LW',
        'player_positions_orig_enc_ST',
        'player_positions_orig_enc_CF'
    ]
)

# Perform hierarchical clustering using Ward's method
Z = linkage(features, method='ward')

# Plot the dendrogram
plt.figure(figsize=(15, 7))
dendrogram(
    Z,
    truncate_mode='level',    # Show only the last p merged clusters
    p=10,                     # Show last 10 levels of the dendrogram
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True
)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# Assign cluster labels
cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')

# Add cluster labels to the DataFrame
preprocessed_df['Cluster'] = cluster_labels

# Save the DataFrame with cluster labels
preprocessed_df.to_csv('players_with_clusters.csv', index=False)

print("Hierarchical clustering completed and 'players_with_clusters.csv' saved successfully!")
