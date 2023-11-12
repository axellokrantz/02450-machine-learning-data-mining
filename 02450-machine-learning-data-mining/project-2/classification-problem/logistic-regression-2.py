import pandas as pd
import numpy as np
from scipy import stats

# Assuming df is your DataFrame and it already has the data
df = pd.DataFrame({
    'Etest_ANN': [0.306878, 0.275132, 0.301587, 0.216931, 0.238095, 0.308511, 0.287234, 0.244681, 0.234043, 0.202128],
    'Etest_Logistic': [0.285714, 0.264550, 0.322751, 0.253968, 0.227513, 0.303191, 0.292553, 0.234043, 0.281915, 0.202128],
    'Etest_Baseline': [0.328042, 0.328042, 0.328042, 0.328042, 0.328042, 0.329787, 0.329787, 0.329787, 0.329787, 0.329787]
})

# Define the pairs of models to compare
pairs = [('Etest_ANN', 'Etest_Logistic'), ('Etest_ANN', 'Etest_Baseline'), ('Etest_Logistic', 'Etest_Baseline')]

for pair in pairs:
    model1, model2 = pair
    # Perform t-test
    t_stat, p_val = stats.ttest_rel(df[model1], df[model2])
    
    # Compute the confidence interval
    confidence_interval = stats.t.interval(0.95, len(df[model1])-1, loc=np.mean(df[model1]-df[model2]), scale=stats.sem(df[model1]-df[model2]))
    
    # Print the results
    print(f"Comparing {model1} against {model2}:")
    print(f"t-statistic: {t_stat}, p-value: {p_val}")
    print(f"95% confidence interval: {confidence_interval}\n")
