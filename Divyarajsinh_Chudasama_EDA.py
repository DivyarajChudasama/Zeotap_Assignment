import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
customers_path = 'customers.csv'
products_path = 'Products.csv'
transactions_path = 'Transactions.csv'

customers_df = pd.read_csv(customers_path)
products_df = pd.read_csv(products_path)
transactions_df = pd.read_csv(transactions_path)

# Display the first few rows
print(customers_df.head(), products_df.head(), transactions_df.head())
# Checking for missing values
missing_values = {
    "Customers": customers_df.isnull().sum(),
    "Products": products_df.isnull().sum(),
    "Transactions": transactions_df.isnull().sum()
}
print(missing_values)
# Checking for duplicates
duplicates = {
    "Customers": customers_df.duplicated().sum(),
    "Products": products_df.duplicated().sum(),
    "Transactions": transactions_df.duplicated().sum()
}
print(duplicates)
# Summary statistics
statistics = {
    "Customers": customers_df.describe(include="all"),
    "Products": products_df.describe(include="all"),
    "Transactions": transactions_df.describe(include="all")
}
print(statistics)
# Merging datasets
transactions_extended = transactions_df.merge(customers_df, on="CustomerID", how="left").merge(products_df, on="ProductID", how="left")
print(transactions_extended.head())
# Total sales by region
region_sales = transactions_extended.groupby("Region")["TotalValue"].sum().sort_values(ascending=False)
print(region_sales)
# Total sales by product category
category_sales = transactions_extended.groupby("Category")["TotalValue"].sum().sort_values(ascending=False)
print(category_sales)
# Analyze signup trends
customers_df["SignupDate"] = pd.to_datetime(customers_df["SignupDate"], format="%d-%m-%Y")
signup_trends = customers_df.groupby(customers_df["SignupDate"].dt.to_period("M")).size()
print(signup_trends)
# Analyzing repeat transactions by customers
repeat_customers = transactions_extended.groupby("CustomerID").size()
repeat_customers_summary = repeat_customers.value_counts().sort_index()

# Top 5 products by sales volume and revenue
top_products_by_volume = transactions_extended.groupby("ProductName")["Quantity"].sum().sort_values(ascending=False).head(5)
top_products_by_revenue = transactions_extended.groupby("ProductName")["TotalValue"].sum().sort_values(ascending=False).head(5)

print(repeat_customers_summary, top_products_by_volume, top_products_by_revenue)

plt.figure(figsize=(10, 6))
region_sales.plot(kind="bar", color='orange', edgecolor='black', title="Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales (USD)")
plt.xticks(rotation=45)
plt.title("Total Sales by Region", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))
category_sales.plot(kind="bar", color='green', edgecolor='black', title="Total Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Sales (USD)")
plt.xticks(rotation=45)
plt.title("Total Sales by Product Category", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))
signup_trends.plot(color='purple', marker='o', linestyle='-', linewidth=2, title="Customer Signups Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Signups")
plt.xticks(rotation=45)
plt.title("Customer Signups Over Time", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

