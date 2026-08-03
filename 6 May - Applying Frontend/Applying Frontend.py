import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

#Load the dataset
dataset = pd.read_csv(r"D:\Machine-Learning\6 May - Applying Frontend\Salary_Data.csv")
# cheak the shape of the dataset
print("Dataset shape:", dataset.shape) # (30,2)

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


y_pred  = regressor.predict(x_test)
print(y_pred)

comparison = pd.DataFrame({'Actual': y_test, 'predicted': y_pred})
print(comparison)

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

model_coef = regressor.coef_
print(model_coef)

model_const = regressor.intercept_
print(model_const)

y_12 = model_coef*12 + model_const
print(y_12)


y_20 = model_coef*20 + model_const
print(y_20)

# mean
dataset.mean() # his will give mean of entire dataset

#median
dataset.median() # this will give median of entire dataframe

#mode
dataset['Salary'].mode() # this will give us mode of that particular column


dataset.var() #this will give variance of entire dataframe

dataset['Salary'].var() # this will give us variance of that particular column

dataset.std() # this will give us standard deviation of that particular column
 
## coefficient of variable(vc)

# for calculating cv we have to  import a library first
from scipy.stats import variation

variation(dataset.values) # this will give cv entire datafram

variation(dataset['Salary']) # this will give us cv of that particular column

# corelation

dataset.corr() # this will give correlation of dataframe

dataset['Salary'].corr(dataset['YearsExperience']) # this will give us correlarion between these
## Skewness

dataset.skew() # this will give skewness of entire dataframe

dataset['Salary'].skew() # this will give us skewness of that particular column


## Standard Error

dataset.sem() # this will give standard error of entire dataframe

dataset['Salary'].sem() # this will give us standard error of that particular column

## Z-score

# for calculating Z-score we have to import a sirst
import scipy.stats as stats

dataset.apply(stats.zscore) # this will give z - score of entire dataframe

stats.zscore(dataset['Salary']) # this will give us z-score of that particular cilumn

## degree of freedom

a = dataset.shape[0] # this will gives us no. of rows
b = dataset.shape[1] # this will give us no.of columns

degree_of_freedom = a-b
print(degree_of_freedom) # this will give us degree of freedom for entire dataset

## ssr
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

## SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(dataset.values) # here dataset.to_numpy
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2 

r_square = 1 - SSR/SST
print(r_square)

# to check overfitting (low bias high variance)
bias = regressor.score(x_train, y_train)
print(bias)

# to check underfitting (high bias low variance)
variance = regressor.score(x_test, y_test)
print(variance)

# deployment in flask & html
# mlops (azur, googlecolab, heroku, kubarnate)

import pickle

#save the trained model to disk
filename = 'linear_regression_model.pkl'

# open a file in write-binary mode and dump the model
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

print("model has been pickled and saved as linear_regression_model.pkl")

import os
os.getcwd()
