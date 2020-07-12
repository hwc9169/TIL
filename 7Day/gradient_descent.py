import numpy as np

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp = x[idx]

        x[idx] = tmp + h
        fx1 = f(x)

        x[idx] = tmp - h
        fx2 = f(x)

        grad[idx] = (fx1-fx2)/(2*h)
        x[idx] = tmp
    return grad

def SGD(f,x,lr=0.01,step_num=100):
    x = x
    for i in range(step_num):
        grad = nume
        
from torch.nn import MSELoss