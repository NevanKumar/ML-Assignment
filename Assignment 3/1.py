import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
file_path = r"C:\Users\KIIT\Desktop\ML\Assignment 3\kmeans - kmeans_blobs.csv"
df = pd.read_csv(file_path)

# Apply K-Means for k=2 and k=3
k_values = [2, 3]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['x1', 'x2']])

    # Plot the clusters
    axes[i].scatter(df['x1'], df['x2'], c=df['cluster'], cmap='viridis', edgecolors='k')
    axes[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    c='red', marker='x', s=100, label='Centroids')
    axes[i].set_title(f'K-Means Clustering (k={k})')
    axes[i].set_xlabel('x1')
    axes[i].set_ylabel('x2')
    axes[i].legend()

plt.tight_layout()
plt.show()