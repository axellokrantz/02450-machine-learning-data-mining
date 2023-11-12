import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV  # Import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import RandomizedSearchCV

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

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'hidden_layer_sizes': [(x,) for x in range(1, 11)],  # Number of hidden units varies from 1 to 10
    'alpha': [10 ** x for x in range(-2, 3)]  # Regularization strength varies from 1e-4 to 1e4
}

# Initialize the MLPRegressor and RandomizedSearchCV
mlp = MLPRegressor(max_iter=1000, random_state=42)
clf = RandomizedSearchCV(mlp, param_dist, cv=KFold(n_splits=10), scoring='neg_mean_squared_error', n_jobs=-1)

# Initialize the DataFrame to store the results
results = pd.DataFrame(columns=['Fold', 'h*', 'λ*', 'E_test_ANN', 'E_test_Ridge', 'Baseline Error'])

# Initialize the DummyRegressor
dummy = DummyRegressor(strategy='mean')
# Perform two-level cross-validation
kf = KFold(n_splits=10)
for i, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the RandomizedSearchCV and find optimal hyperparameters
    clf.fit(X_train, y_train)
    h_star, lambda_star = clf.best_params_['hidden_layer_sizes'][0], clf.best_params_['alpha']
    
    # Compute the test error using optimal hyperparameters for ANN
    mlp_optimal = MLPRegressor(hidden_layer_sizes=(h_star,), alpha=lambda_star, max_iter=1000, random_state=42)
    mlp_optimal.fit(X_train, y_train)
    E_test_ANN = np.mean((y_test - mlp_optimal.predict(X_test))**2)  # Squared loss per observation
    
    # Compute the test error for Ridge regression with optimal alpha (lambda_star)
    ridge_optimal = RidgeCV(alphas=[lambda_star])  # Use RidgeCV with lambda_star as alpha
    ridge_optimal.fit(X_train, y_train)
    E_test_Ridge = np.mean((y_test - ridge_optimal.predict(X_test))**2)  # Squared loss per observation
    
    # Compute the baseline error
    dummy.fit(X_train, y_train)
    baseline_error = np.mean((y_test - dummy.predict(X_test))**2)  # Squared loss per observation
    
    # Append the results to the DataFrame
    results.loc[i] = [i+1, h_star, lambda_star, E_test_ANN, E_test_Ridge, baseline_error]

print(results)