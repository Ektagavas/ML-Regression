# -*- coding: utf-8 -*-
"""MultivariateRegression.ipynb


# Multivariate Regression using Gradient Descent variants

### Import Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""### Read Dataset"""

my_data = pd.read_csv('https://raw.githubusercontent.com/Tan-Moy/medium_articles/master/art2_multivariate_linear_regression/home.txt', names=["size", "bedroom", "price"])
my_data.head()

"""### Normalize data"""

my_data = (my_data - my_data.mean()) / my_data.std()
my_data.head()

"""### Create X, Y and theta matrices"""

X = my_data.iloc[:,0:2]
Y = my_data.iloc[:,2:3].values
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis = 1)
theta = np.zeros([1,3])

"""### Hyper parameters

#### Learning rate
"""

alpha = 0.01

"""#### No of iterations of gradient descent"""

itr = 50

"""### Cost function"""

def compute_cost(X, Y, theta):
    total = np.power(Y - (X.dot(theta.T)),2)
    return (np.sum(total)/(len(X)))

"""### Initial cost value"""

cost = compute_cost(X, Y, theta)
cost

"""### Gradient Descent function"""

def gradient_descent(X, Y, theta, alpha, initial, final):
    
    Xb = X[initial:final,:]
    Yb = Y[initial:final,:]
    
    derv_theta = np.zeros(len(theta.T))
    for i in range(1,len(theta.T)):
        total = -((Yb - (Xb.dot(theta.T)))*Xb)
        derv_theta[i] =(2 * np.sum(total))/len(Xb)
    theta = theta - (alpha * derv_theta)
    return theta

"""### Run Gradient Descent"""

def run_gradient_desc(batch_size):
    costarr = np.zeros(itr)
    theta = np.zeros([1,3])
    batches = (int)(len(X) / batch_size)
    if (len(X) % batch_size)!=0:
        batches += 1
    for i in range(itr):
        for batch in range(0,batches):
            initial = (batch * batch_size)
            final = (initial + batch_size)
            if final > len(X):
                final = len(X)
            theta = gradient_descent(X, Y, theta, alpha, initial, final)
        costarr[i] = compute_cost(X, Y, theta)
    #print(cost)
    return theta, costarr

"""## Batch Gradient Descent i.e. batch_size = n"""

theta, costarr1 = run_gradient_desc(len(X))

"""### Visualize batch gradient descent"""

x = np.arange(1,itr+1)
plt.plot(x, costarr1, 'b-')
plt.xlabel("Iterations", fontsize = 12)
plt.ylabel("Cost", fontsize = 12)
plt.title("Cost vs Iterations", fontsize = 14)

"""## Stochastic Gradient Descent i.e. batch_size = 1"""

theta, costarr2 = run_gradient_desc(1)

"""### Visualize stochastic gradient descent"""

x = np.arange(1,itr+1)
plt.plot(x, costarr2, 'r-')
plt.xlabel("Iterations", fontsize = 12)
plt.ylabel("Cost", fontsize = 12)
plt.title("Cost vs Iterations", fontsize = 14)

"""## Mini Batch Gradient Descent with batch_size = 16"""

theta, costarr3 = run_gradient_desc(16)

"""### Visualize mini batch gradient descent"""

x = np.arange(1,itr+1)
plt.plot(costarr3, 'g-')
plt.xlabel("Iterations", fontsize = 12)
plt.ylabel("Cost", fontsize = 12)
plt.title("Cost vs Iterations", fontsize = 14)

"""### Final cost value"""

cost = compute_cost(X, Y, theta)
cost
