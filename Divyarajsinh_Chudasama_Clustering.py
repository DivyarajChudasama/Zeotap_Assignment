import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from sklearn.decomposition import PCA
customers_path = 'customers.csv'
products_path = 'Products.csv'
transactions_path = 'Transactions.csv'

customers_df = pd.read_csv(customers_path)
products_df = pd.read_csv(products_path)
transactions_df = pd.read_csv(transactions_path)
# Merge Customers.csv and Transactions.csv
customer_transactions = transactions_df.groupby("CustomerID").agg({
    "TotalValue": ["sum", "mean"],    # Total and average spending
    "Quantity": "sum",               # Total quantity
    "ProductID": "nunique"           # Product diversity
}).reset_index()

# Rename columns
customer_transactions.columns = ["CustomerID", "TotalSpending", "AvgSpending", "TotalQuantity", "UniqueProducts"]

# Merge with Customers.csv
segmentation_data = customer_transactions.merge(customers_df, on="CustomerID", how="left")

# One-hot encode the region
region_dummies = pd.get_dummies(segmentation_data["Region"], prefix="Region")
segmentation_data = pd.concat([segmentation_data, region_dummies], axis=1)

# Drop unnecessary columns
segmentation_data = segmentation_data.drop(columns=["CustomerName", "Region", "SignupDate"])

# Normalize the data for clustering
scaler = StandardScaler()
scaled_segmentation_data = scaler.fit_transform(segmentation_data.drop(columns=["CustomerID"]))



# Perform KMeans clustering for different cluster counts (2 to 10)
db_scores = []
kmeans_models = {}

for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_segmentation_data)
    db_index = davies_bouldin_score(scaled_segmentation_data, cluster_labels)
    db_scores.append((n_clusters, db_index))
    kmeans_models[n_clusters] = (kmeans, cluster_labels)

# Choose the best cluster count based on DB Index (lower is better)
optimal_clusters = sorted(db_scores, key=lambda x: x[1])[0]
best_kmeans, best_labels = kmeans_models[optimal_clusters[0]]


# Reduce dimensions for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_segmentation_data)

# Plot clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters[0]):
    cluster_points = pca_data[best_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
    
plt.title(f"Customer Clusters (K={optimal_clusters[0]})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid()
plt.show()

# Add cluster labels to the segmentation data
segmentation_data["Cluster"] = best_labels

# Save clustering results to CSV
clustering_csv_path = "Customer_Clusters.csv"
segmentation_data.to_csv(clustering_csv_path, index=False)

print(f"Clustering results saved at: {clustering_csv_path}")