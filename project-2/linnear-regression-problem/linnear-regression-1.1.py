import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Split the standardized data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Ridge Regression model with the chosen lambda value (lambda = 100)
ridge_model = Ridge(alpha=100)
ridge_model.fit(X_scaled, y)

# Make predictions on the testing data with Ridge Regression model
y_pred_ridge = ridge_model.predict(X_test)

# Calculate MAE and MSE for the Ridge Regression model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"Ridge Regression - Mean Absolute Error: {mae_ridge}")
print(f"Ridge Regression - Mean Squared Error: {mse_ridge}")

# Create a scatterplot of actual vs. predicted values for Ridge Regression
plt.scatter(X_test, y_test, alpha=0.5, label="Actual Values")
plt.scatter(X_test, y_pred_ridge, alpha=0.5, label="Predicted Values (Ridge)", color='red')
plt.xlabel("Oscore (Centered and Standardized)")
plt.ylabel("Escore")
plt.title("Actual vs. Predicted Values (Ridge Regression)")
plt.legend()
plt.grid(True)
plt.show()

# Example: Predicting 'Escore' for a new 'Oscore'
new_oscore = np.array([1])  # Replace 'some_value' with the actual 'Oscore' value you want to predict
new_oscore_scaled = scaler.transform(new_oscore.reshape(-1, 1))
predicted_escore = ridge_model.predict(new_oscore_scaled)
print("Predicted Escore with Ridge Regression:", predicted_escore)
