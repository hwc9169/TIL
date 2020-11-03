import numpy as np


def MSE(x,y):
    return 0.5*np.sum((y-x)**2)

y =np.array([0,0,1,0,0,0,0,0,0,0,])
x = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
cost = MSE(x,y)
print(cost)

x = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])
cost = MSE(x,y)
print(cost)

