import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('drug_consumption.data', delimiter=',')

# Extract attribute_names.
attribute_names = df.columns.tolist()
# Extract class_names, where each entry is a person.
y = df.iloc[:, 0].values

# Processing of data. Removal of unecessary prefix 'CL'.
for col in df.columns[13:32]:
    df[col] = df[col].str.replace('CL', '').astype(int)

# Insert data into matrix.
X = df.iloc[:, 1:]

# Get number of data objects.
N = X.shape[0]

# Calculate the mean of each feature (column)
feature_means = np.mean(X, axis=0)

# Subtract the means from the data matrix to mean-center it
X_centered = X - feature_means

# Calculate Z-Score of X_centered
Z = zscore(X_centered)

# Identify outliers
# outliers = np.where(np.abs(Z) > 3)

Z = np.array(Z)
#Z = Z[(np.abs(Z) <= 4).all(axis=1)]

# Create histograms
for i in range(Z.shape[1]):
    plt.figure(i)
    plt.hist(Z[:,i], bins=30)
    plt.title(attribute_names[i+1])  # i+1 because we removed the first column (y)
plt.show()

# Create a DataFrame from the Z-scores
Z_df = pd.DataFrame(Z, columns=attribute_names[1:])

# Outliers identified:
# People who claim to have used semer.
# People with ethnicity asian / black. 

#Z_df = Z_df[Z_df['Semer'] < 0]

# Create box plot
plt.figure(figsize=(10,6))
plt.boxplot(Z_df.values, vert=False, labels=Z_df.columns)
plt.title('Box plot of all attributes')
plt.show()
# Calculate correlation matrix
corr = Z_df.corr()

# Create a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm')

plt.title('Heatmap of Correlation Matrix')
plt.show()
