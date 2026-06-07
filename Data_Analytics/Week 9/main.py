import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# ==========================
# 1. Load Data
# ==========================

file_path = "Fitness_App_User_Data.xlsx"

df = pd.read_excel("Fitness_App_User_Data.xlsx")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ==========================
# 2. Basic Data Cleaning
# ==========================

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

# Remove User_ID (not useful for clustering)
if 'User_ID' in df.columns:
    df = df.drop('User_ID', axis=1)

print("\nData after cleaning:")
print(df.info())

# ==========================
# 3. Encode Categorical Data
# ==========================

label_encoder = LabelEncoder()

categorical_columns = ['Gender', 'Subscription_Type']

for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# ==========================
# 4. Select Features
# ==========================

# Exclude target column if present
features = df.drop(columns=['Churned'])

# ==========================
# 5. Scale Features
# ==========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)

# ==========================
# 6. Elbow Method
# ==========================

inertia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# ==========================
# 7. Apply K-Means
# ==========================

optimal_clusters = 3

kmeans = KMeans(
    n_clusters=optimal_clusters,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

# ==========================
# 8. Cluster Summary
# ==========================

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

print("\nCluster Statistics:")
print(df.groupby('Cluster').mean())

# ==========================
# 9. Visualize Clusters
# ==========================

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df['Steps_per_Day'],
    df['Avg_Session_Duration_Min'],
    c=df['Cluster']
)

plt.xlabel('Steps Per Day')
plt.ylabel('Avg Session Duration (Min)')
plt.title('Fitness User Clusters')

plt.colorbar(scatter, label='Cluster')
plt.show()

# ==========================
# 10. Save Results
# ==========================

output_file = "Fitness_App_User_Data_Clustered.xlsx"

df.to_excel(output_file, index=False)

print(f"\nClustered data saved as: {output_file}")