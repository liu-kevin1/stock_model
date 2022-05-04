from flask import Flask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = "C:\\Users\\orgwi\\OneDrive\\Desktop\\stock_model_repo\\stock_model"

BACK_COUNT = 30

def reformat_file(filePath, backCount):
    file = open(filePath, "r")
    
    # Store the text to write later
    newText = ""
    
    lines = file.read().split("\n")
    
    #lines = lines[0:10]
    
    # Reformat the first line, the header
    firstLine = lines[0]
    features = firstLine.split(",")
    newLine = firstLine
    for back in range(1, backCount+1):
        for f in features:
            newLine += "," + str(back) + "_" + f
    newText = newText + newLine + "\n"
    
    # Reformat the rest of the lines, the data
    for i in range(1, len(lines)):
        line = lines[i]
        newLine = ""
        for back in range(1, backCount+2):
            if (back != 1):
                newLine += ","
            if i <= back:
                newLine += line
            else:
                newLine += lines[i-back]
        newText = newText + newLine + "\n"
        #print(newLine)
    
    # [:-4] for cutting off the .txt
    newPath = filePath[:-4] + "_reformatted.txt"
    newFile = open(newPath, "w+")
    newFile.write(newText)
    newFile.close()
    return newPath
                
# stocks_data = pd.read_csv(data_path)
# print(stocks_data.head())

# reformat_file(data_path, 30)

def print_path_contents(p):
    for dirname, _, filenames in os.walk(p):
        for filename in filenames:
            print(os.path.join(dirname, filename))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
data_path = path + "\\data\\Stocks\\a.us.txt"
# data_path = path + "\\a.us copy_reformatted.txt"

data_path = reformat_file(data_path, BACK_COUNT)

pd.options.display.max_columns = None
pd.options.display.max_rows = None

stocks_data = pd.read_csv(data_path)
print(stocks_data.head())
y = stocks_data.Close

# Create X (After completing the exercise, you can return to modify this line!)
features = ["Open", "High", "Low", "Volume"]

backCount = BACK_COUNT

newFeatures = []
for f in features:
    newFeatures.append(f)
    
for i in range(1, backCount+1):
    for f in features:
        newFeatures.append(str(i) + "_" + f)

features = newFeatures

# Select columns corresponding to features, and preview the data
X = stocks_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1) #, test_size=0.8)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print(val_X.head())
print(val_y.head())
print(stocks_data.head())

plt.figure(figsize=(10,10))
plt.scatter(val_y, rf_val_predictions, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(rf_val_predictions), max(val_y))
p2 = min(min(rf_val_predictions), min(val_y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# plt.plot()
# plt.show()
input("Press ENTER to continue")

# Load the data, and separate the target
data_path = path + "\\data\\Stocks\\aa.us.txt"
# data_path = path + "\\a.us copy_reformatted.txt"

data_path = reformat_file(data_path, BACK_COUNT)

pd.options.display.max_columns = None
pd.options.display.max_rows = None

stocks_data = pd.read_csv(data_path)
print(stocks_data.head())
y = stocks_data.Close

# Create X (After completing the exercise, you can return to modify this line!)
features = ["Open", "High", "Low", "Volume"]

backCount = BACK_COUNT

newFeatures = []
for f in features:
    newFeatures.append(f)
    
for i in range(1, backCount+1):
    for f in features:
        newFeatures.append(str(i) + "_" + f)

features = newFeatures

# Select columns corresponding to features, and preview the data
X = stocks_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1) #, test_size=0.8)

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

input("Press ENTER to exit")