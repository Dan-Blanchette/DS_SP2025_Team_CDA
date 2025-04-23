# ----------------
# @file     ml_analysis.py
# @date     4/14/25
# @author   Jordan Reed
# @class    CS 579 Data Science
# @brief    init ml analysis with multiple models
# ----------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------
# load data
# ----------
df = pd.read_csv("/home/jreed/data-science-group-project/cleaned_aqi_hospitalization_data_for_VGboost_RF.csv")
# print(df.columns)

# ----------
# data wrangling
# ----------
outcomes = 'Value'
features = [
    'StateFIPS', 
    'CountyFIPS', 
    'Year',
    'Good Days', 
    'Moderate Days', 'Unhealthy for Sensitive Groups Days', 
    'Unhealthy Days', 'Very Unhealthy Days', #'Hazardous Days', 
    'Max AQI', '90th Percentile AQI', 
    'Median AQI',
    ]

# normalize days with how many days recorded
df['Good Days'] = df['Good Days']/df['Days with AQI']
df['Moderate Days'] = df['Moderate Days']/df['Days with AQI']
df['Unhealthy for Sensitive Groups Days'] = df['Unhealthy for Sensitive Groups Days']/df['Days with AQI']
df['Unhealthy Days'] = df['Unhealthy Days']/df['Days with AQI']
df['Very Unhealthy Days'] = df['Very Unhealthy Days']/df['Days with AQI']
df['Hazardous Days'] = df['Hazardous Days']/df['Days with AQI']

# ---------
# set up train test sets
# ---------
train_features = df[features]
train_outcomes = df[outcomes]

print(f'Max hospitalizations: {np.max(train_outcomes)}')
print(f'Min hospitalizations: {np.min(train_outcomes)}')
print(f'Mean hospitalizations: {np.mean(train_outcomes)}')

training_features, test_features, training_outcomes, test_outcomes = train_test_split(train_features, train_outcomes, test_size=0.3)
training_features, val_features, training_outcomes, val_outcomes = train_test_split(train_features, train_outcomes, test_size=.1)

# -----------
# set up models
# -----------
models = [
    RandomForestRegressor(max_features='sqrt', max_depth=20),
    BaggingRegressor(n_estimators=100, max_samples=700, max_features=9),
    GradientBoostingRegressor(learning_rate=.8),
    LinearRegression(),
]
model_names = ['Random Forest', 'Bagging', 'Boosting', 'Linear Regression']

for i,model in enumerate(models):
    # ------------
    # train model
    # ------------
    model.fit(training_features, training_outcomes)

    print(f'\n----{model_names[i]} model trained')
    # ------------
    # score model accuracy
    # ------------
    mean_accuracy = model.score(training_features, training_outcomes)
    print(f'mean training acc: {mean_accuracy*100:.2f} %')

    mean_test_acc = model.score(test_features, test_outcomes)
    print(f'mean testing acc: {mean_test_acc*100:.2f} %')

    mean_val_acc = model.score(val_features, val_outcomes)
    print(f'mean validation acc: {mean_val_acc*100:.2f} %')

    # -------
    # calculate errors
    # -------

    predictions_train = model.predict(train_features)
    predictions_test = model.predict(test_features)
    predictions_val = model.predict(val_features)

    mae_train = mean_absolute_error(train_outcomes, predictions_train)
    print(f'MAE Train: {mae_train:.2f}')
    mae_test = mean_absolute_error(test_outcomes, predictions_test)
    print(f'MAE Test: {mae_test:.2f}')
    mae_val = mean_absolute_error(val_outcomes, predictions_val)
    print(f'MAE: {mae_val:.2f}')

    # ---------
    # calculate r2
    # ---------
    r2_val_train = r2_score(train_outcomes, predictions_train)
    print(f'R2 val train: {r2_val_train:.2f}')
    r2_val_test = r2_score(test_outcomes, predictions_test)
    print(f'R2 val test: {r2_val_test:.2f}')
    r2_val_val = r2_score(val_outcomes, predictions_val)
    print(f'R2 val valid: {r2_val_val:.2f}')


    # ------------
    # visualize predictions
    # ------------

    # # showing error
    # error = test_outcomes - predictions
    # plt.scatter(predictions, error)
    # plt.axhline(0, color='red', linestyle='--')
    # plt.xlabel("Predicted")
    # plt.ylabel("Error")
    # plt.title(f"Error vs predicted - {model_names[i]}")
    # plt.show()

    # find correlation?
    ax = plt.subplot()
    ax.spines['right'].set_color((.9,.9,.9))
    ax.spines['top'].set_color((.9,.9,.9))
    plt.scatter(test_outcomes, predictions_test, color=[.5, .5, .8])
    plt.plot([min(test_outcomes),max(test_outcomes)], [min(test_outcomes),max(test_outcomes)], color=[0, 0, .5], linestyle='solid') # optimal line
    plt.xlabel("Actual", style="italic")
    plt.ylabel("Predicted", style="italic")
    plt.title(f"Prediction vs Actual Hospitalizations- {model_names[i]}")
    plt.show()
