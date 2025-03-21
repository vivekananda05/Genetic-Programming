# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:53:10 2025

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
from sklearn.ensemble import RandomForestRegressor
from gplearn.genetic import SymbolicRegressor

train_data = pd.read_excel("train_data.xlsx")
x_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

#scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
# qt = QuantileTransformer(n_quantiles=45, output_distribution='normal')
# x_train = qt.fit_transform(x_train)



est_gp = SymbolicRegressor(population_size=10000,
                            generations=100,
                            tournament_size=10,
                            stopping_criteria=0.01,
                            p_crossover=0.7,  
                            p_subtree_mutation=0.1,
                            p_hoist_mutation=0.01, 
                            p_point_mutation=0.1,
                            #p_point_replace=0.05,
                            max_samples=0.9, 
                            verbose=1,
                            parsimony_coefficient=0.01, 
                            random_state=42,
                            n_jobs=-1)

est_gp.fit(x_train, y_train)

print(est_gp._program)

with open("symbolic_regressor_model.pkl", "wb") as f:
    pickle.dump(est_gp, f)

test_data = pd.read_excel("test_data.xlsx")
x_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]
x_test = scaler.fit_transform(x_test)
#x_test = qt.fit_transform(x_test)
#y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)).ravel()

y_pred = est_gp.predict(x_test)
y_pred_train = est_gp.predict(x_train)


min_val = min(min(y_test), min(y_pred), min(y_pred_train))
max_val = max(max(y_test), max(y_pred), max(y_pred_train))

score_gp_test = est_gp.score(x_test, y_test)
score_gp_train = est_gp.score(x_train, y_train)
print(score_gp_test)
print(score_gp_train)

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
plt.title('Comparison of Metrics for Train and Test Data in Genetic Programming')
plt.legend()

for i, (t_val, tst_val) in enumerate(zip(train_metrics, test_metrics)):
    plt.text(x[i] - width/2, t_val + 0.01, f'{t_val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
    plt.text(x[i] + width/2, tst_val + 0.01, f'{tst_val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_train, y_pred_train, color='green', alpha=0.6, label='Training Data')
plt.scatter(y_test, y_pred, color='blue', alpha=0.8, label='Test Data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values for Genetic Programming')
plt.legend()
plt.show()