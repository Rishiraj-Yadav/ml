import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Load dataset
df = pd.read_csv("customers.csv")

# Select spending features
X = df.drop(columns=["Channel", "Region"])

# Standardize the data
X_scaled = StandardScaler().fit_transform(X)

# Calculate Euclidean distances and Ward linkage
Z = linkage(X_scaled, method="ward", metric="euclidean")

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Index")
plt.ylabel("Euclidean Distance")
plt.show()

# Create 3 customer clusters
df["Cluster"] = fcluster(Z, t=3, criterion="maxclust")

print("\nCustomer Clusters:")
print(df[["Fresh", "Milk", "Grocery", "Cluster"]])

# Visualize clusters
sns.scatterplot(
    data=df, x="Fresh", y="Milk",
    hue="Cluster", palette="Set1", s=100
)
plt.title("Wholesale Customer Clusters")
plt.show()