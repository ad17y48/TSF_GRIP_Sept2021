# Author: Aditya Bagla
# The Spark Foundation, Graduate Rotational Program September 2021
# Data Science and Business Analytics Task
# Technical TASK 1:  Prediction using Supervised ML (Level - Beginner)
# Task Description: In this task, we will predict the percentage of marks that a student
# is expected to score based upon the number of hours
# they studied. This is a simple linear regression task as it involves just two variables.


# Importing required libraries
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Importing and Reading the dataset from remote link
data = pd.read_csv('http://bit.ly/w-data')
print(data)
print("data imported successfully!")


# Plotting the distribution of score
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.show()


# dividing the data into "attributes" (inputs) and "labels" (outputs)
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Training the model
reg = LinearRegression()
reg.fit(x_train.reshape(-1, 1), y_train)
print("Trained!!")


# Plotting the regression line
line = reg.coef_ * x + reg.intercept_


# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line, color='Black')
plt.show()


# Testing data - In Hours
print(x_test)
# Predicting the scores
y_predict = reg.predict(x_test)


# Comparing Actual vs Predicted
data = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
print(data)


# Estimating the Training Data and Test Data Score
print("Training score:", reg.score(x_train, y_train))
print("Testing score:", reg.score(x_test, y_test))


# Plotting the line graph to depict the difference between actual and predicted value.
data.plot(kind='line', figsize=(8, 5))
plt.grid(which='major', linewidth='0.5', color='black')
plt.grid(which='major', linewidth='0.5', color='black')
plt.show()


# Testing the model.
hours = 9.25
test_data = np.array([hours])
test_data = test_data.reshape(-1, 1)
own_predict = reg.predict(test_data)
print("Hours = {}".format(hours))
print("Predicted Score = {}".format(own_predict[0]))


# Checking the efficiency of model
# This is the final step to evaluate the performance of an algorithm.
# This step is particularly important to compare how well different algorithms
# perform on a given dataset
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))
print('Root mean squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
