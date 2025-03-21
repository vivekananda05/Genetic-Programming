# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:22:37 2024

@author: vivek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

train_data = pd.read_excel("train_data.xlsx")
x_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
#qt = QuantileTransformer(n_quantiles=10, output_distribution='normal')
#x_train = qt.fit_transform(x_train)

alpha_range = np.logspace(0, 2, 5)

ridge = Ridge(max_iter=10000, random_state=12)
grid_search = GridSearchCV(estimator=ridge, param_grid={'alpha': alpha_range}, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)


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
plt.title('Cross-Validation Score vs. Regularization parameter (λ) for Ridge Regression')
plt.show()

print(f"Best Alpha: {best_alpha}")

coefs = []
for alpha in alpha_range:
    ridge.alpha = alpha
    ridge.fit(x_train, y_train)
    coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alpha_range, coefs)
plt.xscale('log')
plt.xlabel('λ')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Path')
plt.show()

test_data = pd.read_excel("test_data.xlsx")
x_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]
x_test = scaler.fit_transform(x_test)
#x_test = qt.fit_transform(x_test)

ridge_final = Ridge(alpha=best_alpha, max_iter=10000, random_state=12)
ridge_final.fit(x_train, y_train)
y_pred = ridge_final.predict(x_test)
y_pred_train = ridge_final.predict(x_train)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(mse_test, r2_test, rmse_test, mae_test)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
print(mse_train, r2_train, rmse_train, mae_train)

ridge_coefficients = ridge_final.coef_
print(ridge_coefficients)
metrics = ['MAE','RMSE', 'R²']
train_metrics = [mae_train, rmse_train, r2_train]
test_metrics = [mae_test, rmse_test, r2_test]
x = np.arange(len(metrics))  
width = 0.35  

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, train_metrics, width, label='Train', color='green', alpha=0.7)
plt.bar(x + width/2, test_metrics, width, label='Test', color='blue', alpha=0.7)

plt.xticks(x, metrics)
plt.ylabel('Metric Value')
plt.title('Comparison of Metrics for Train and Test Data in Ridge Regression')
plt.legend()

for i, (t_val, tst_val) in enumerate(zip(train_metrics, test_metrics)):
    plt.text(x[i] - width/2, t_val + 0.01, f'{t_val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
    plt.text(x[i] + width/2, tst_val + 0.01, f'{tst_val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
plt.tight_layout()
plt.show()

min_val = min(min(y_test), min(y_pred), min(y_pred_train))
max_val = max(max(y_test), max(y_pred), max(y_pred_train))

plt.figure(figsize=(8, 8))
plt.scatter(y_train, y_pred_train, color='green', alpha=0.6, label='Training Data')
plt.scatter(y_test, y_pred, color='blue', alpha=0.8, label='Test Data')
plt.plot([min(y_train), max(y_train)], [min(y_train),max(y_train)],color='red', linestyle='-')
# plt.text(min(y_test), max(y_test),
#          f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}', 
#          fontsize=10, color='blue', verticalalignment='top')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values for Ridge Regression')
plt.legend()
plt.show()

res = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, res, color='black', alpha=0.6, label='residuals')
plt.axhline(y=0, color='red', linestyle='-', label='zero error line')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual plot on test data for Ridge Regression')
plt.show()

