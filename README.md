# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THIRUMALAI V
RegisterNumber: 212223040229

*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the CSV file using the correct separator (tab character in this case)
df = pd.read_csv("student_scores123.csv", sep="\t")
print(df.head())  # Check the structure of the DataFrame

# Assuming the last column is the target variable
x = df.iloc[:, :-1].values  # Features (Hours)
y = df.iloc[:, -1].values   # Target variable (Scores)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color="orangered", s=60)
plt.plot(x_train, regressor.predict(x_train), color="darkviolet", linewidth=4)
plt.title("Hours vs Scores (Training set)", fontsize=24)
plt.xlabel("Hours", fontsize=18)
plt.ylabel("Scores", fontsize=18)
plt.show()
plt.scatter(x_test, y_test, color="seagreen", s=60)
plt.plot(x_test, regressor.predict(x_test), color="cyan", linewidth=4)
plt.title("Hours vs Scores (Test set)", fontsize=24)
plt.xlabel("Hours", fontsize=18)
plt.ylabel("Scores", fontsize=18)
plt.show()

mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print("RMSE = ", rmse)


```

## Output:

![image](https://github.com/user-attachments/assets/3ee164a5-1166-45cd-98ef-6d09f82b02c0)
![image](https://github.com/user-attachments/assets/00f7fc36-9192-43a3-84ca-6bf2480eac29)
![image](https://github.com/user-attachments/assets/b2c49e42-b725-4982-9d01-010428c7709e)
![image](https://github.com/user-attachments/assets/c4a230fd-2ec4-4add-acab-1097b96b217d)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
