import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load the data
df = pd.read_csv('drug_consumption.data', delimiter=',')

# Extract attribute_names.
attribute_names = df.columns.tolist()

# Processing of data. Removal of unnecessary prefix 'CL'.
for col in df.columns[13:32]:
    df[col] = df[col].str.replace('CL', '').astype(int)

# Extract data and calculate Z-scores
X = df.iloc[:, 1:]
X_centered = X - np.mean(X, axis=0)
Z = zscore(X_centered)

# Create a DataFrame from the Z-scores
Z_df = pd.DataFrame(Z, columns=attribute_names[1:])

# Identify outliers based on the 'Semer' attribute
Z_df = Z_df[Z_df['Semer'] < 0]

# Calculate the number of rows and columns for subplots
num_attributes = Z_df.shape[1]
num_rows = (num_attributes - 1) // 4 + 1
num_cols = min(4, num_attributes)

# Create histograms in a single plot
plt.figure(figsize=(15, 10))

for i in range(num_attributes):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.hist(Z_df.iloc[:, i], bins=30)
    plt.title(attribute_names[i+1])  # i+1 because we removed the first column (y)

plt.tight_layout()
plt.show()
