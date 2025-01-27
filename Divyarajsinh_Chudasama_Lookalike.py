import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
customers_path = 'customers.csv'
products_path = 'Products.csv'
transactions_path = 'Transactions.csv'

customers_df = pd.read_csv(customers_path)
products_df = pd.read_csv(products_path)
transactions_df = pd.read_csv(transactions_path)
# Merge datasets to combine customer, transaction, and product information
transactions_extended = transactions_df.merge(customers_df, on="CustomerID", how="left").merge(products_df, on="ProductID", how="left")

# Feature Engineering: Aggregate transaction data for each customer
enhanced_features = transactions_extended.groupby("CustomerID").agg({
    "TotalValue": ["sum", "mean", "std"],    # Spending metrics
    "Quantity": ["sum", "mean"],            # Quantity metrics
    "ProductID": "nunique",                 # Product diversity
    "Category": "nunique",                  # Category diversity
}).reset_index()

# Rename columns for clarity
enhanced_features.columns = [
    "CustomerID",
    "TotalSpending",
    "AvgSpending",
    "SpendingStdDev",
    "TotalQuantity",
    "AvgQuantity",
    "UniqueProducts",
    "UniqueCategories"
]

# Add Region information as categorical features (one-hot encoding)
region_dummies = pd.get_dummies(customers_df.set_index("CustomerID")["Region"], prefix="Region")
enhanced_features = enhanced_features.merge(region_dummies, left_on="CustomerID", right_index=True, how="left")

# Handle missing values (fill NaN with 0)
enhanced_features.fillna(0, inplace=True)

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(enhanced_features.drop(columns=["CustomerID"]))


# Apply PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(scaled_features)

# Fit a Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=4, metric="euclidean")  # 3 neighbors + self
knn_model.fit(pca_features)

# Find the nearest neighbors for each customer
distances, indices = knn_model.kneighbors(pca_features)
# Create a dictionary to store recommendations
lookalike_map = {}

# Convert distances to similarity scores
for idx, customer_id in enumerate(enhanced_features["CustomerID"].values):
    similar_indices = indices[idx][1:4]  # Exclude the customer itself
    similar_customers = enhanced_features.iloc[similar_indices]["CustomerID"].values
    similar_scores = 1 / (1 + distances[idx][1:4])  # Convert distances to similarity scores
    lookalike_map[customer_id] = list(zip(similar_customers, similar_scores))

# Extract recommendations for the first 20 customers
lookalike_list = [
    {"CustomerID": cust, "Lookalikes": lookalikes}
    for cust, lookalikes in lookalike_map.items()
    if cust in customers_df["CustomerID"][:20].values
]

# Convert to a DataFrame and save to CSV
lookalike_df = pd.DataFrame(lookalike_list)
lookalike_csv_path = "Lookalike.csv"
lookalike_df.to_csv(lookalike_csv_path, index=False)
print(f"Lookalike CSV saved at: {lookalike_csv_path}")
