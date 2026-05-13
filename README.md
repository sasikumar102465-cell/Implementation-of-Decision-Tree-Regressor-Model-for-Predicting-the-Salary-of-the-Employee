# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the placement dataset, convert the target variable (Placed / Not Placed) into binary form, select SSC percentage and MBA percentage as features, and normalize the feature values.

2.Add a bias term to the input data, initialize model parameters (theta), learning rate, and number of iterations.

3.For each iteration, compute predictions using the sigmoid function, calculate the cost function, compute gradients, and update parameters using gradient descent.

4.Predict the placement status, calculate the accuracy of the model, and plot the cost versus iterations graph. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SASIKUMAR S
RegisterNumber: 212225100046

#Ex 09 - Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ------------------------------
# Step 1: Sample dataset
# ------------------------------

df = pd.read_csv('Salary.csv')

# ------------------------------
# Step 2: Split features and target
# ------------------------------
X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)

# ------------------------------
# Step 3: Create Decision Tree Regressor
# ------------------------------
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# ------------------------------
# Step 4: Predict salary for the dataset or new levels
# ------------------------------
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

# Example: predict salary for a new employee at level 6.5
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

# ------------------------------
# Step 5: Visualize the results (High-resolution curve)
# ------------------------------
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
*/
```

## Output:
<img width="815" height="78" alt="image" src="https://github.com/user-attachments/assets/7abb9f97-3106-4194-8eb3-9c6fcd1cbd87" />

<img width="733" height="584" alt="image" src="https://github.com/user-attachments/assets/2b961bd6-30bc-4955-8b70-eb949f743800" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
