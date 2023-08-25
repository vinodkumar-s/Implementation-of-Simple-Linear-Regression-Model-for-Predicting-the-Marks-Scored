# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
Program to implement the simple linear regression model For predicting the marks scored.

Developed by: s.vinod kumar
RegisterNumber:  212222240116

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values

y_test

#graph plot for training data

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()


mse=mean_squared_error(y_test,y_pred)
print('MSE= ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE =',rmse)
```

## Output:

## df.head()

![simple linear regression model for predicting the marks scored](/Screenshot%202023-08-25%20112729.png)

## df.tail()
![OUTPUT](/Screenshot%202023-08-25%20112737.png)

## Array value of X
![OUTPUT](/Screenshot%202023-08-25%20113120.png)

## Array value of Y
![output](/Screenshot%202023-08-25%20113134.png)

## Values of Y prediction

![output](/Screenshot%202023-08-25%20113404.png)

## Values of Y test


![output](/Screenshot%202023-08-25%20113409.png)

## Training Set Graph


![output](/Screenshot%202023-08-25%20113425.png)

##  Test Set Graph

![output](/Screenshot%202023-08-25%20114413.png)

## Values of MSE, MAE and RMSE

![output](/image-2.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
