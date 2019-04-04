# -*- coding: utf-8 -*-
"""LinearRegression.ipynb

### Import libraries
"""

import numpy as np
import matplotlib.pyplot as plt

"""### Visualize Dataset"""

my_data = np.genfromtxt('https://raw.githubusercontent.com/kaustubholpadkar/Linear_Regression-Gradient_Descent-Octave/master/data.csv', delimiter=',')

xdata = my_data[:,0]
ydata = my_data[:,1]

plt.plot(xdata,ydata, 'bo')
plt.xlabel("Study Hours", fontsize = 12)
plt.ylabel("Expected Marks", fontsize = 12)

"""### Parameters"""

b = 0
m = 0

alpha = 0.0001  # Learning rate

"""### Cost function"""

def compute_cost(m,b):
    total = 0;
    for i in range(0,100):
        cost = ydata[i] - (m * xdata[i] + b)
        total += cost * cost
    return total/len(xdata)

cost = compute_cost(m,b)
print(cost)

"""### Gradient Descent"""

def gradient_descent(m, b, alpha):
    derv_lm = 0
    derv_lb = 0
    cost = compute_cost(m,b)
    for i in range(0,100):
        j = (ydata[i] - ((m * xdata[i]) + b)) * (-xdata[i])
        derv_lm += j
    derv_lm = (2 * derv_lm)/len(xdata)
    
    for i in range(0,100):
        j = (ydata[i] - ((m * xdata[i]) + b)) * (-1)
        derv_lb += j
    derv_lb = (2 * derv_lb)/len(xdata)
    
    m = m - (alpha * derv_lm)
    b = b - (alpha * derv_lb)
    new_cost = compute_cost(m,b)
    print(new_cost)
    #if new_cost > cost:
        #print("stop")
    return m,b

max_itr = 100

def run_gradient_desc():
    m = 0
    b = 0
    for i in range(1, max_itr):
        m,b = gradient_descent(m, b, alpha)
    return m, b

m,b = run_gradient_desc()

"""### Plot regression line"""

x = np.arange(25,75)
y = [m * xi + b for xi in x]

nm,nb = np.polyfit(xdata, ydata, 1)
nx = np.arange(25,75)
ny = [nm * xi + nb for xi in x]

plt.plot(xdata, ydata, 'bo', x , y, 'r-', nx, ny, 'g-')
plt.legend(loc = 'upper left', labels = ['data', 'myline', 'numpyline'])
plt.xlabel("Study Hours", fontsize = 12)
plt.ylabel("Expected Marks", fontsize = 12)



