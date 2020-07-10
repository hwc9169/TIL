import numpy as np
import torch

def step_func(x):
    y = x>0
    return y.astype(np.int)

def AND(x1 ,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(x*w)+b
    if tmp >= 0:
        return 1
    else:
        return 0

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(x*w)+b
    if tmp >=0:
        return 1
    else:
        return 0

def OR(x1,x2):
    x = np.array([x1, x2])
    w = np.array([0.5,0.5])
    b = -0.3
    tmp = np.sum(x * w) + b
    if tmp >= 0:
        return 1
    else:
        return 0

def XOR(x1,x2):
    tmp1 = NAND(x1,x2)
    tmp2 = OR(x1,x2)
    tmp = AND(tmp1,tmp2)
    return tmp

print(step_func(1))

