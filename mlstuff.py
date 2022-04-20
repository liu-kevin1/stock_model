from flask import Flask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = "C:\\Users\\orgwi\\OneDrive\\Desktop\\stock_model_repo\\stock_model"

def print_path_contents(p):
    for dirname, _, filenames in os.walk(p):
        for filename in filenames:
            print(os.path.join(dirname, filename))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
data_path = path + "\\data\\Stocks\\a.us.txt"

stocks_data = pd.read_csv(data_path)
print(stocks_data.head())
y = stocks_data.Close

# Create X (After completing the exercise, you can return to modify this line!)
features = ["Open", "High", "Low", "Volume"]

# Select columns corresponding to features, and preview the data
X = stocks_data[features]
for line in X["Open"]:
    print(line)
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print(val_X.head())
print(val_y.head())
