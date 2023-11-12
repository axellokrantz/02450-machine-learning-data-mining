import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('drug_consumption.data', delimiter=',')

# Processing of data. Removal of unnecessary prefix 'CL' and conversion to integers.
for col in df.columns[13:32]:
    df[col] = df[col].str.replace('CL', '').str.strip().astype(int)

# Select the feature and target variable
X = df[['Oscore']]  # 'Oscore' is the feature
y = df['Escore'].values  # 'Escore' is the target variable

# Mean-center the predictor variable
X_mean = X.mean()
X_centered = X - X_mean

# Standardize the predictor variable to have mean 0 and std 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_centered)

# Create a list of different numbers of hidden layers to test
hidden_layers = [(x,) for x in range(1, 11)]  # Number of hidden units varies from 1 to 10

# Lists to store number of hidden layers and corresponding MSE
hidden_layer_values = []
mse_values = []

for hidden_layer in hidden_layers:
    # Create the ANN model with the specified number of hidden layers
    ann_model = MLPRegressor(hidden_layer_sizes=hidden_layer, max_iter=1000, random_state=42)

    # Perform K-fold cross-validation to estimate the generalization error
    scores = cross_val_score(ann_model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    mse = -scores.mean()  # Take the negative mean squared error

    hidden_layer_values.append(hidden_layer[0])
    mse_values.append(mse)

    print(f"Number of Hidden Layers: {hidden_layer[0]}")
    print(f"Mean Squared Error (MSE) with 10-fold Cross-Validation: {mse:.4f}\n")

# Create a plot of number of hidden layers vs. MSE
plt.figure()
plt.plot(hidden_layer_values, mse_values, marker='o')
plt.title('Number of Hidden Layers vs. Mean Squared Error (MSE)')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.show()
