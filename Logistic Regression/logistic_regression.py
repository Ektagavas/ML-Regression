# -*- coding: utf-8 -*-
"""Logistic Regression.ipynb

# Logistic Regression
## IRIS classification

### Import Libraries
"""

# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize

"""### Load Training Data"""

dataset = load_iris()

"""### Separate Data into X and Y"""

X = dataset.data[:,:2]
Y = (dataset.target != 0) * 1
Y = Y.reshape(Y.shape[0], 1)

"""### Visualize the Data"""

fig = plt.figure()
plt.scatter(X[0:50,0], X[0:50,1], color='blue')
plt.scatter(X[50:,0], X[50:,1], color='red')
plt.legend(loc='upper left', labels=['0', '1'])
fig.suptitle('Iris Data', fontsize=18)
plt.xlabel('Sepal Length', fontsize=14)
plt.ylabel('Sepal Width', fontsize=14)
plt.show()

"""### Normalize Data"""

X = normalize(X, axis=0)

"""### Sigmoid function"""

def sigmoid (z):
  return 1 / (1 + np.exp(-z))

"""### Hypothesis function"""

def h (W, b, X):
  Z = np.matmul(X, W) + b
  A = sigmoid(Z)
  return A

"""### Loss function"""

def loss (W, b, X, Y):
  A = h(W, b, X)
  return -np.average((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))

"""### Finding the Gradients"""

def gradient (W, b, X, Y) :
  A = h(W, b, X)
  dW = np.matmul(X.T, (A - Y))
  db = np.matmul(np.ones((1, X.shape[0])), (A - Y))
  return (dW, db)

"""### Gradient Descent function"""

def gradient_descent (W, b, X, Y, learning_rate, max_iteration, gap) :
  cost = np.zeros(max_iteration)
  for i in range(max_iteration) :
    dW, db = gradient(W, b, X, Y)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    cost[i] = loss (W, b, X, Y)
    if i % gap == 0 :
      print ('iteration : ', i, ' loss : ', loss (W, b, X, Y)) 
  return W, b, cost

"""### Stochastic Gradient Descent function"""

def stochastic_gradient_descent (W, b, X, Y, learning_rate, max_iteration, gap) :
  cost = np.zeros(max_iteration)
  for i in range(max_iteration) :
    for j in range(X.shape[0]):
      dW, db = gradient(W, b, X, Y)
      W = W - learning_rate * dW
      b = b - learning_rate * db
    
    cost[i] = loss (W, b, X, Y)
    if i % gap == 0 :
      print ('iteration : ', i, ' loss : ', loss (W, b, X, Y)) 
  return W, b, cost

"""### Hyper parameters"""

learning_rate = 0.09
max_iteration = 2000
gap = 200

"""### Parameters"""

W = np.zeros((X.shape[1], 1))  
b = np.zeros((1, 1))

"""### Train Model"""

W, b, cost = stochastic_gradient_descent (W, b, X, Y, learning_rate, max_iteration, gap)

"""### Cost vs Iteration Plots"""

#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(max_iteration), cost, 'r') 
ax.legend(loc='upper right', labels=['stochastic gradient descent'])
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  
plt.show()

"""### Calculate Slope, Intercept and Points for Separating line"""

m = - (W[0, 0] / W[1, 0])
c = - (b[0, 0] / W[1, 0])


x = np.arange(0.055, 0.115, 0.005)
y = [m * x_i + c for x_i in x]

"""### Visualize the Data"""

fig = plt.figure()
plt.scatter(X[0:50,0], X[0:50,1], color='blue')
plt.scatter(X[50:,0], X[50:,1], color='red')
plt.plot(x,y, color='green')
plt.legend(loc='upper left', labels=['Separating Line', '0', '1'])
fig.suptitle('Iris Data', fontsize=18)
plt.xlabel('Sepal Length', fontsize=14)
plt.ylabel('Sepal Width', fontsize=14)
plt.show()

