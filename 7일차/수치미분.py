import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def f3(x):
    y = x>0
    return y.astype(int)

def f2(x):
    return 2*x**2

def f1(x):
    return 0.01*(x**2) + 0.1*x

x = [[np.arange(-20.0, 20.0 , 0.1)],[np.arange(-20.0, 20.0 , 0.1)]]
plt.plot(x,f4(x))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
