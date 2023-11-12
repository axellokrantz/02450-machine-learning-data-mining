import numpy as np
from scipy import stats

# Given data
E_test_ANN = np.array([0.885984, 0.708664, 0.727574, 0.689305, 1.018264, 1.024005, 0.956177, 0.957258, 1.366449, 1.239402])
E_test_Ridge = np.array([0.885385, 0.711209, 0.733328, 0.677560, 1.018857, 1.017696, 0.961494, 0.960400, 1.364267, 1.126202])
Baseline_Error = np.array([1.005151, 0.786252, 0.712230, 0.764717, 1.162917, 1.127193, 1.042915, 1.037440, 1.283714, 1.042619])

# Paired t-tests
t_statistic_ann_ridge, p_value_ann_ridge = stats.ttest_rel(E_test_ANN, E_test_Ridge)
t_statistic_ann_baseline, p_value_ann_baseline = stats.ttest_rel(E_test_ANN, Baseline_Error)
t_statistic_ridge_baseline, p_value_ridge_baseline = stats.ttest_rel(E_test_Ridge, Baseline_Error)

print(f"ANN vs Ridge: t-statistic = {t_statistic_ann_ridge}, p-value = {p_value_ann_ridge}")
print(f"ANN vs Baseline: t-statistic = {t_statistic_ann_baseline}, p-value = {p_value_ann_baseline}")
print(f"Ridge vs Baseline: t-statistic = {t_statistic_ridge_baseline}, p-value = {p_value_ridge_baseline}")

# Confidence intervals
confidence_level = 0.95
degrees_freedom = len(E_test_ANN) - 1

sample_mean_ann_ridge = np.mean(E_test_ANN - E_test_Ridge)
sample_standard_error_ann_ridge = stats.sem(E_test_ANN - E_test_Ridge)
confidence_interval_ann_ridge = stats.t.interval(confidence_level,
                                                 degrees_freedom,
                                                 sample_mean_ann_ridge,
                                                 sample_standard_error_ann_ridge)

sample_mean_ann_baseline = np.mean(E_test_ANN - Baseline_Error)
sample_standard_error_ann_baseline = stats.sem(E_test_ANN - Baseline_Error)
confidence_interval_ann_baseline = stats.t.interval(confidence_level,
                                                    degrees_freedom,
                                                    sample_mean_ann_baseline,
                                                    sample_standard_error_ann_baseline)

sample_mean_ridge_baseline = np.mean(E_test_Ridge - Baseline_Error)
sample_standard_error_ridge_baseline = stats.sem(E_test_Ridge - Baseline_Error)
confidence_interval_ridge_baseline = stats.t.interval(confidence_level,
                                                      degrees_freedom,
                                                      sample_mean_ridge_baseline,
                                                      sample_standard_error_ridge_baseline)

print(f"ANN vs Ridge: confidence interval = {confidence_interval_ann_ridge}")
print(f"ANN vs Baseline: confidence interval = {confidence_interval_ann_baseline}")
print(f"Ridge vs Baseline: confidence interval = {confidence_interval_ridge_baseline}")
