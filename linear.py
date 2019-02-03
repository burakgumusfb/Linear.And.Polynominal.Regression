import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error

rg = LinearRegression()
dataset = pd.read_csv('data.csv')
dataset.drop('id', axis=1, inplace=True)

x = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)
rg.fit(xTrain,yTrain)
yPrediction = rg.predict(xTest)

plot.scatter(xTest, yTest, color = 'red')
plot.plot(xTest, yPrediction, color = 'blue')
plot.title('Salary vs Experience (Training set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

print (rg.score(xTest,yTest))


