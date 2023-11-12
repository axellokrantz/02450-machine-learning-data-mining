import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from toolbox_02450.similarity import similarity

df = pd.read_csv('drug_consumption.data', delimiter=',')

# Extract attribute_names.
attribute_names = df.columns.tolist()
# Extract class_names, where each entry is a person.
y = df.iloc[:, 1].values

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

# Similarity measures: 'SMC', 'Jaccard', 'ExtendedJaccard', 'Cosine', 'Correlation' 
similarity_measures = ['SMC', 'Jaccard', 'ExtendedJaccard', 'Cosine', 'Correlation']

# Calculate similarity using the function for each measure
similarity_matrices = {}
for measure in similarity_measures:
    similarity_matrices[measure] = similarity(X_centered, X_centered, measure)
    
    
