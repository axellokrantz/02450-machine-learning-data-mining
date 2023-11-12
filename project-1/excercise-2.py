# Step 1: Import data from .data file.
# Step 2: Extract attribute_names.
# Step 3: Extract class_names.
# Step 4: Process data.
# Step 5: Insert data in N x M matrix X.
# Step 6: 

# Step 4: Insert data in N x M matrix. 
#   N = rows (participants), M = collumns (attributes)
# Step 5: Adjust X matrix, remove class index into separate array y.
# Step 6: Mean centering of data X into Y.
# Step 7: PCA by computing SVD of Y.
# Step 8: Plot.

import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('drug_consumption.data', delimiter=',')

# Extract attribute_names.
attribute_names = df.columns.tolist()

# Extract class_names, where each entry is a person.
y = df.iloc[:, 0].values

# Processing of data. Removal of unnecessary prefix 'CL'.
for col in df.columns[13:32]:
    df[col] = df[col].str.replace('CL', '').astype(int)

# Insert data into matrix.
X = df.iloc[:, 1:]

# Standardize the data by the standard deviation prior to PCA analysis.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA by computing SVD of Y.
U, S, V = svd(X_scaled, full_matrices=False)

# Compute variance explained by principal components
explained_variance_ratio = (S ** 2) / np.sum(S ** 2)
cumulative_variance = np.cumsum(explained_variance_ratio)

# Calculate loadings for each principal component
loadings = V * np.sqrt(S[:, np.newaxis])

# Create a DataFrame to store the loadings and associate them with feature names
loading_df = pd.DataFrame(loadings, columns=df.columns[1:])

# Select the first three principal components for plotting
pcs_to_plot = [0, 1, 2]

# Create a bar chart to visualize the coefficients of PC0, PC1, and PC2 in different colors
plt.figure(figsize=(12, 6))
colors = ['r', 'g', 'b']

for i, component_to_visualize in enumerate(pcs_to_plot):
    plt.bar(
        np.arange(len(loading_df.columns)) + i * 0.2,
        loading_df.iloc[component_to_visualize, :],
        width=0.2,
        color=colors[i],
        label=f'PC{component_to_visualize}'
    )

plt.xlabel('Features')
plt.ylabel('Component Coefficients')
plt.title('PCA Component Coefficients for PC0, PC1, and PC2')
plt.xticks(np.arange(len(loading_df.columns)), loading_df.columns, rotation=90)
plt.legend()
plt.grid(True)
plt.show()

# Plot the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components (SVD)')
plt.grid(True)
plt.show()

# Project data onto principal components
PCs = np.dot(X_scaled, V.T)

# Plot the data projected onto the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(PCs[:, 0], PCs[:, 1])
plt.xlabel('PC0')
plt.ylabel('PC1')
plt.title('Data Projected onto First Two Principal Components')
plt.grid(True)
plt.show()

# Find the number of components needed to explain 95% of the variance
num_components = np.where(cumulative_variance > 0.90)[0][0] + 1

print(f'{num_components} components explain 95% of the variance.')
