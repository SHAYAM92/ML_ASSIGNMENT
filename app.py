# K-means Clustering
# Initial Clusters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the earthquake dataset
data = pd.read_csv('earthquake_data.csv')  # Update with your dataset path

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Latitude', 'Longitude']])

# Step 2: Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
initial_clusters = kmeans.fit_predict(data_scaled)

# Plot initial clusters
plt.scatter(data['Latitude'], data['Longitude'], c=initial_clusters, cmap='viridis', s=50, alpha=0.6)
plt.title("Initial Clusters using K-means")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

# Final Clusters along with Epoch Size
# Step 2: Apply K-means clustering to get final clusters
kmeans = KMeans(n_clusters=3, random_state=42, max_iter=10)
final_clusters = kmeans.fit_predict(data_scaled)
epochs = kmeans.n_iter_

# Display results
print("Final Clusters (first 10):", final_clusters[:10])
print("Number of epochs (iterations) to convergence:", epochs)

# Plot final clusters
plt.scatter(data['Latitude'], data['Longitude'], c=final_clusters, cmap='viridis', s=50, alpha=0.6)
plt.title(f"Final Clusters using K-means - Epochs: {epochs}")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

# Final Clusters along with Error Rate
# Calculate error rate (inertia)
error_rate = kmeans.inertia_
print("Error Rate (Inertia):", error_rate)

# Plot final clusters with error rate in title
plt.scatter(data['Latitude'], data['Longitude'], c=final_clusters, cmap='viridis', s=50, alpha=0.6)
plt.title(f"Final Clusters using K-means - Error Rate: {error_rate:.2f}")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

# 2. Mean Shift Clustering
# Initial Clusters
from sklearn.cluster import MeanShift

# Step 1: Load the earthquake dataset
data = pd.read_csv('earthquake_data.csv')  # Update with your dataset path

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Latitude', 'Longitude']])

# Step 2: Apply Mean Shift clustering
mean_shift = MeanShift(bandwidth=2)
initial_clusters = mean_shift.fit_predict(data_scaled)

# Plot initial clusters
plt.scatter(data['Latitude'], data['Longitude'], c=initial_clusters, cmap='viridis', s=50, alpha=0.6)
plt.title("Initial Clusters using Mean Shift")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

#Final Clusters along with Iteration Count
# Step 2: Apply Mean Shift clustering to get final clusters
final_clusters = mean_shift.fit_predict(data_scaled)

# Display number of clusters and centroids
n_clusters = len(np.unique(final_clusters))
final_centroids = mean_shift.cluster_centers_
print("Number of Final Clusters:", n_clusters)
print("Final Centroids:\n", final_centroids)

# Plot final clusters
plt.scatter(data['Latitude'], data['Longitude'], c=final_clusters, cmap='viridis', s=50, alpha=0.6)
plt.title(f"Final Clusters using Mean Shift - Clusters: {n_clusters}")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

#Final Clusters along with Error Rate
# Calculate error rate (WCSS)
error_rate = 0
for i in range(len(final_centroids)):
    cluster_points = data_scaled[final_clusters == i]
    error_rate += np.sum((cluster_points - final_centroids[i]) ** 2)

print("Error Rate (Within-Cluster Sum of Squares):", error_rate)

# Plot final clusters with error rate in title
plt.scatter(data['Latitude'], data['Longitude'], c=final_clusters, cmap='viridis', s=50, alpha=0.6)
plt.title(f"Final Clusters using Mean Shift - Error Rate: {error_rate:.2f}")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

