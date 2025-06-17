# In this notebook, we will perform clustering analysis on the preprocessed mortality data from Malaysia. The goal is to identify patterns and group similar data points based on their features.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the model-ready data
df = pd.read_csv("../data/processed/malaysia_mortality_data_model.csv")

# Let's filter the data to a specific year (2021) and sex ('Persons') to create a stable, comparable snapshot for clustering. This removes temporal and gender noise.
cluster_data = df[(df["Year"] == 2021) & (df["Sex_Persons"] == 1)].copy()

# Pivot the data to create 'mortality profiles'. Each disease becomes a vector representing its mortality pattern across all age groups.
mortality_profiles = cluster_data.pivot_table(
    index="Disease_L2", columns="Age Group", values="Mortality Count"
).fillna(0)

print("Mortality Profiles for Clustering:")
print(mortality_profiles.head())

# %%
# Scaling is essential for K-Means, as it prevents age groups with high absolute mortality counts (e.g., 70+) from dominating the clustering algorithm.

# Standardize the features (age groups) to have equal weight.
scaler = StandardScaler()
scaled_profiles = scaler.fit_transform(mortality_profiles)

# %%
# The Elbow Method helps determine a suitable number of clusters.
# The "elbow" point indicates where adding more clusters yields diminishing returns in variance reduction.
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_profiles)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_range)
plt.grid(True)
plt.show()

# The plot shows a distinct elbow around k=3, but the curve continues to bend until k=5.
# Let's choose k=5 to provide more granular clusters for a richer analysis.

# %%
# Let's choose k=5 based on the elbow plot and project requirements.
OPTIMAL_K = 5
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_profiles)

# Add the cluster labels back to our original profiles DataFrame
mortality_profiles["Cluster"] = clusters

# Analyze the characteristics of each cluster by looking at the mean profile
cluster_analysis = mortality_profiles.groupby("Cluster").mean()

print(cluster_analysis)
# %%
# Visualize the cluster centroids
cluster_analysis.T.plot(
    kind="bar",
    figsize=(15, 7),
    title="Mortality Count by Age Group for Each Cluster",
)
plt.ylabel("Average Mortality Count")
plt.xlabel("Age Group")
plt.xticks(rotation=45)
plt.show()

# %%
# Let's see which diseases fall into which cluster
for i in range(OPTIMAL_K):
    print(f"\n--- Cluster {i} Diseases ---")
    print(mortality_profiles[mortality_profiles["Cluster"] == i].index.tolist())
# --- Interpretation of Clustering Results ---
#
# The K-Means algorithm successfully segmented the 25 disease categories into 5 distinct clusters based on their mortality patterns across different age groups.
# Each cluster represents a unique epidemiological profile, providing valuable insights into how different types of diseases impact the Malaysian population.

# --- Cluster 0: Age-Accelerated Chronic Diseases ---
# Diseases: ['Malignant neoplasms', 'Respiratory Infectious']
# Profile: This cluster shows low mortality in younger age groups, which begins to rise significantly after age 50 and accelerates dramatically in the 70+ group.
# It represents major chronic diseases and severe infections that are strongly correlated with aging and accumulated health risks.

# --- Cluster 1: Broad-Spectrum & Low-Prevalence Diseases ---
# Diseases: ['Diabetes mellitus', 'Digestive diseases', 'Endocrine, blood, immune disorders', etc.]
# Profile: This is a "catch-all" cluster for diseases with relatively low but persistent mortality across a wide range of age groups.
# While individually less impactful than diseases in other clusters, their collective burden is significant and affects the population throughout the lifecycle.

# --- Cluster 2: Dominant Geriatric Diseases ---
# Diseases: ['Cardiovascular diseases']
# Profile: This cluster is defined by a single, dominant cause of death. Its mortality profile is characterized by an exponential increase with age,
# becoming the leading cause of death by a large margin in the elderly population (60+). This highlights its status as a primary public health challenge in an aging nation.

# --- Cluster 3: Mid-Life & Youth-Risk Diseases ---
# Diseases: ['Unintentional injuries']
# Profile: This cluster exhibits a distinct pattern with mortality peaking in younger and middle-aged adult groups (15-49), then declining in older age.
# This profile is characteristic of external causes of death like accidents, which disproportionately affect younger, more active populations.

# --- Cluster 4: Infant and Neonatal Diseases ---
# Diseases: ['Congenital anomalies', 'Neonatal conditions']
# Profile: This cluster is sharply defined, with mortality almost exclusively concentrated in the 0-4 age group.
# It clearly identifies perinatal and congenital conditions that are the primary drivers of infant mortality, distinct from all other disease patterns.

# %%
# --- Hierarchical Clustering Analysis ---
from scipy.cluster.hierarchy import dendrogram, linkage

# Use the 'ward' method which minimizes the variance of the clusters being merged.
linked = linkage(scaled_profiles, method="ward")

# Plot the dendrogram
plt.figure(figsize=(15, 10))
dendrogram(
    linked,
    orientation="top",
    labels=mortality_profiles.index.tolist(),
    distance_sort="descending",
    show_leaf_counts=True,
)
plt.title("Hierarchical Clustering Dendrogram of Mortality Profiles")
plt.xlabel("Disease Category")
plt.ylabel("Distance (Ward)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# The dendrogram provides a detailed, hierarchical view of the relationships between disease mortality profiles, complementing the findings from K-Means.
#
# The highest-level split in the tree separates 'Neonatal conditions' and 'Congenital anomalies' from all other diseases.
# This is the most significant distinction, confirming that infant mortality patterns (K-Means Cluster 4) are fundamentally different from all other mortality profiles.
#
# The dendrogram also reveals subtle relationships.
# For example, it shows that 'Respiratory Infectious' diseases are grouped more closely with chronic conditions like neoplasms than with other infectious diseases, likely due to their severe impact on older, more vulnerable populations.
