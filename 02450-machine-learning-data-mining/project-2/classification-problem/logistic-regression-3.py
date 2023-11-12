import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# Load the dataset
df = pd.read_csv('drug_consumption.data', delimiter=',')

# Processing of data. Removal of unnecessary prefix 'CL' and conversion to integers.
for col in df.columns[13:32]:
    df[col] = df[col].str.replace('CL', '').str.strip().astype(int)

# Define the target variable 'Cannabis' as a binary classification problem
df['Cannabis'] = df['Cannabis'].apply(lambda x: 0 if x <= 1 else 1)

# Select the feature and target variable
X = df[['Nscore', 'Escore', 'Oscore', 'Cscore']]
y = df['Cannabis']

# Standardize the predictor variables to have mean 0 and std 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the standardized data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the hyperparameter space
hyperparameters = {'C': np.logspace(-4, 4, 20)}

# Train the Logistic Regression model with GridSearchCV
logistic_model = LogisticRegression(penalty='l2')
clf = GridSearchCV(logistic_model, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(X_train, y_train)

# Make predictions on the testing data with Logistic Regression model
y_pred_logistic = best_model.predict(X_test)

# Calculate accuracy for the Logistic Regression model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

# Calculate error for the Logistic Regression model
error_logistic = mean_squared_error(y_test, y_pred_logistic)

print(f"Logistic Regression - Accuracy: {accuracy_logistic}")
print(f"Logistic Regression - Error: {error_logistic}")

# Print the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_logistic)
print(f"Confusion Matrix:\n{confusion_mat}")

# Print the best lambda (C) value
print('Best C:', best_model.best_estimator_.get_params()['C'])

# Plot the CV error vs. lambda value
plt.figure(figsize=(8, 6))
plt.semilogx(hyperparameters['C'], clf.cv_results_['mean_test_score'])
plt.xlabel('Lambda')
plt.ylabel('Mean CV Score')
plt.title('CV Error vs. Lambda Value')
plt.show()
