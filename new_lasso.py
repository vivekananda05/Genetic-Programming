# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:43:36 2025

@author: vivek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import shap
from SALib.sample import saltelli
from SALib.analyze import sobol

train_data = pd.read_excel("train_data.xlsx")
x_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

test_data = pd.read_excel("test_data.xlsx")
x_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
with open("symbolic_regressor_model1.pkl", "rb") as f:
    est_gp_loaded = pickle.load(f)
    
y_pred_train = est_gp_loaded.predict(x_train)
y_pred_test = est_gp_loaded.predict(x_test)

x_train_df = pd.DataFrame(x_train, columns=train_data.columns[1:])
x_train_df["y_pred_train"] = y_pred_train
x_test_df = pd.DataFrame(x_test, columns=train_data.columns[1:])
x_test_df["y_pred_test"] = y_pred_test

# plt.figure(figsize=(8, 6))
# plt.scatter(y_train, y_pred_train, color='blue', alpha=0.6, label='Predictions')
# plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='dashed', linewidth=2, label='Perfect Fit')
# plt.xlabel("Actual Values (y_train)")
# plt.ylabel("Predicted Values (y_pred_train)")
# plt.title("Actual vs. Predicted Values")
# plt.legend()
# plt.show()

x_train_new = x_train_df.to_numpy()
x_test_new = x_test_df.to_numpy()
scaler = StandardScaler()
x_train_new = scaler.fit_transform(x_train_new)
x_test_new = scaler.fit_transform(x_test_new)

alpha_range = np.logspace(-1, 0, 10)

lasso = Lasso(max_iter=10000, random_state=12)

grid_search = GridSearchCV(estimator=lasso, param_grid={'alpha': alpha_range}, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train_new, y_train)

results = grid_search.cv_results_
alphas = results['param_alpha'].data.astype(float)
mean_mse = -results['mean_test_score']
std_mse = -results['std_test_score']
best_alpha = grid_search.best_params_['alpha']

plt.figure(figsize=(10, 6))
plt.plot(np.log10(alphas), mean_mse, color='red', label='Mean MSE')
plt.fill_between(np.log10(alphas), mean_mse - std_mse, mean_mse + std_mse, color='gray', alpha=0.3, label='Standard Deviation')
plt.axvline(np.log10(grid_search.best_params_['alpha']), color='blue', linestyle='--', label=f'Best Alpha: {best_alpha}')
plt.xlabel('log(λ)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Cross-Validation Score vs. Regularization parameter (λ) for Lasso Regression')
plt.show()

print(f"Best Alpha: {best_alpha}")

coefs = []
for alpha in alpha_range:
    lasso.alpha = alpha
    lasso.fit(x_train_new, y_train)
    coefs.append(lasso.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alpha_range, coefs)
plt.xscale('log')
plt.xlabel('λ')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Path')
plt.show()

lasso_final = Lasso(alpha=best_alpha, max_iter=10000, random_state=12)
lasso_final.fit(x_train_new, y_train)
y_pred_test_final = lasso_final.predict(x_test_new)
y_pred_train_final = lasso_final.predict(x_train_new)

explainer = shap.LinearExplainer(lasso_final, x_train_df)
shap_values = explainer(x_train_df)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, x_train_df)
plt.title("Feature Importance Plot For Lasso")
plt.show()

min_val = min(min(y_test), min(y_pred_test_final), min(y_pred_train_final))
max_val = max(max(y_test), max(y_pred_test_final), max(y_pred_train_final))


mse_test = mean_squared_error(y_test, y_pred_test_final)
r2_test = r2_score(y_test, y_pred_test_final)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test_final)
print(mse_test, r2_test, rmse_test, mae_test)
mse_train = mean_squared_error(y_train, y_pred_train_final)
r2_train = r2_score(y_train, y_pred_train_final)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_pred_train_final)
print(mse_train, r2_train, rmse_train, mae_train)

metrics = ['MAE', 'RMSE', 'R²']
train_metrics = [mae_train, rmse_train, r2_train]
test_metrics = [mae_test, rmse_test, r2_test]
x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, train_metrics, width, label='Train', color='green', alpha=0.7)
plt.bar(x + width/2, test_metrics, width, label='Test', color='blue', alpha=0.7)

plt.xticks(x, metrics)
plt.ylabel('Metric Value')
plt.title('Comparison of Metrics for Train and Test Data in Lasso with Genetic Programming')
plt.legend()

for i, (t_val, tst_val) in enumerate(zip(train_metrics, test_metrics)):
    plt.text(x[i] - width/2, t_val + 0.01, f'{t_val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
    plt.text(x[i] + width/2, tst_val + 0.01, f'{tst_val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_train, y_pred_train_final, color='green', alpha=0.6, label='Training Data')
plt.scatter(y_test, y_pred_test_final, color='blue', alpha=0.8, label='Test Data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values for Lasso with Genetic Programming')
plt.legend()
plt.show()


# Sensitivity Analysis
num_features = x_train_new.shape[1]
feature_names = x_train_df.columns.tolist()

problem = {
    'num_vars': num_features,
    'names': feature_names,
    'bounds': [[x_train_new[:, i].min(), x_train_new[:, i].max()] for i in range(num_features)]
}

# Generate Sobol samples
N = 1024  # Must be a power of 2 for better convergence
param_values = saltelli.sample(problem, N)

# Ensure correct shape
param_values = param_values[:, :num_features]

# Predict using Ridge model for Sensitivity Analysis
Y_sensitivity = lasso_final.predict(param_values)

# Perform Sobol sensitivity analysis
Si = sobol.analyze(problem, Y_sensitivity)

# Convert results to a DataFrame
sensitivity_df = pd.DataFrame({
    "Feature": feature_names,
    "First-order Index (S1)": Si['S1'],
    "Total-order Index (ST)": Si['ST']
})


# Save results to CSV for further analysis
sensitivity_df.to_csv("lasso_sensitivity_analysis.csv", index=False)

plt.figure(figsize=(60, 20))
plt.barh(feature_names, Si['ST'], color='skyblue', edgecolor='black')
plt.xlabel("Total Sensitivity Index (ST)")
plt.ylabel("Feature")
plt.title("Feature Sensitivity Analysis (Sobol Indices) using Lasso model")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()