from turtle import back
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = "C:\\Users\\orgwi\\OneDrive\\Desktop\\stock_model_repo\\stock_model"

# Load the data, and separate the target
# data_path = path + "\\data\\Stocks\\a.us.txt"
data_path = path + "\\a.us copy.txt"

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
    newFile = open(filePath[:-4] + "_reformatted.txt", "w+")
    newFile.write(newText)
                
stocks_data = pd.read_csv(data_path)
print(stocks_data.head())

reformat_file(data_path, 30)
input("Press ENTER to exit")