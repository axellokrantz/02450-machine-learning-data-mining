import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
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

# Create a list of different λ (alpha) values to test
alphas = [0.01, 0.1, 1, 10, 100, 1000]

# Lists to store lambda values and corresponding MSE
lambda_values = []
mse_values = []

for alpha in alphas:
    # Create the Ridge Regression model with the specified λ
    ridge_model = Ridge(alpha=alpha)

    # Perform K-fold cross-validation to estimate the generalization error
    scores = cross_val_score(ridge_model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    mse = -scores.mean()  # Take the negative mean squared error

    lambda_values.append(alpha)
    mse_values.append(mse)

    print(f"Lambda (α): {alpha}")
    print(f"Mean Squared Error (MSE) with 10-fold Cross-Validation: {mse:.4f}\n")

# Create a plot of lambda values vs. MSE
plt.figure()
plt.plot(lambda_values, mse_values, marker='o')
plt.title('Lambda (α) vs. Mean Squared Error (MSE)')
plt.xlabel('Lambda (α)')
plt.ylabel('Mean Squared Error (MSE)')
plt.xscale('log')  # Using a logarithmic scale for lambda values
plt.grid(True)
plt.show()
