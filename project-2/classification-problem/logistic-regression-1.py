import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv('drug_consumption.data', delimiter=',')

# Processing of data. Removal of unnecessary prefix 'CL' and conversion to integers.
for col in df.columns[13:32]:
    df[col] = df[col].str.replace('CL', '').str.strip().astype(int)

# Define the target variable 'Cannabis' as a binary classification problem
df['Cannabis'] = df['Cannabis'].apply(lambda x: 0 if x <= 1 else 1)

# Select the feature and target variable
X = df[['Nscore', 'Escore', 'Oscore', 'Cscore']]  # 'Nscore', 'Escore', 'Oscore', 'Cscore' are the features
y = df['Cannabis']  # 'Cannabis' is the target variable

# Standardize the predictor variables to have mean 0 and std 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the baseline model
baseline = DummyClassifier(strategy='most_frequent')

# Define the logistic regression model
logistic = LogisticRegression()

# Define the ANN model
ann = MLPClassifier(max_iter=1000)

# Define the grid search parameters
param_grid_ann = {'hidden_layer_sizes': [(i,) for i in range(1, 11)]}
param_grid_logistic = {'C': [10**i for i in range(-5, 4)]}

# Create Grid Search
grid_ann = GridSearchCV(estimator=ann, param_grid=param_grid_ann, n_jobs=-1)
grid_logistic = GridSearchCV(estimator=logistic, param_grid=param_grid_logistic, n_jobs=-1)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=10)
results = []

fold = 1  # Initialize fold counter
for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the models
    baseline.fit(X_train, y_train)
    grid_ann.fit(X_train, y_train)
    grid_logistic.fit(X_train, y_train)

    # Predict the results
    y_pred_baseline = baseline.predict(X_test)
    y_pred_ann = grid_ann.predict(X_test)
    y_pred_logistic = grid_logistic.predict(X_test)

    # Compute the error rates
    Etest_Baseline = (y_test != y_pred_baseline).sum() / len(y_test)
    Etest_ANN = (y_test != y_pred_ann).sum() / len(y_test)
    Etest_Logistic = (y_test != y_pred_logistic).sum() / len(y_test)

    # Append the results
    results.append({
        'Fold': fold,
        'Etest_ANN': Etest_ANN,
        'h*': grid_ann.best_params_['hidden_layer_sizes'][0],
        'Etest_Logistic': Etest_Logistic,
        'λ*': grid_logistic.best_params_['C'],
        'Etest_Baseline': Etest_Baseline
    })

    fold += 1  # Increment fold counter

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)
print(results_df)
