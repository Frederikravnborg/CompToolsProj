# Import necessary libraries for clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('players_preprocessed.csv')

# Compute the linkage matrix using Ward's method
Z = linkage(df, method='ward')

# Plot the dendrogram to visualize the clusters
plt.figure(figsize=(15, 7))
dendrogram(
    Z,
    truncate_mode='level',  # show only the last p merged clusters
    p=10,  # show last 10 levels of the dendrogram
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True
)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# Determine the number of clusters from the dendrogram
num_clusters = 5

# Assign cluster labels to each player
cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')

# Add the cluster labels to the original DataFrame
df['Cluster'] = cluster_labels

# Save the DataFrame with cluster labels
df.to_csv('players_with_clusters.csv', index=False)

# (Optional) Visualize the cluster distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=df, palette='viridis')
plt.title('Number of Players in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# (Optional) Analyze clusters by aggregating key statistics
cluster_summary = df.groupby('Cluster').agg({
    'age': 'mean',
    'height_cm': 'mean',
    'weight_kg': 'mean',
    'pace': 'mean',
    'shooting': 'mean',
    'passing': 'mean',
    'dribbling': 'mean',
    'defending': 'mean',
    'physic': 'mean',
    'skill_moves': 'mean',
    'weak_foot': 'mean',
    'mentality_vision': 'mean',
    'mentality_composure': 'mean',
    'mentality_aggression': 'mean',
    'mentality_interceptions': 'mean',
    'mentality_positioning': 'mean',
    'international_reputation': 'mean'
}).reset_index()

print("Cluster Summary:")
print(cluster_summary)
