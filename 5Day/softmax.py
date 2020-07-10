import numpy as np

def softmax(x):
    c = np.max(x)
    exp = np.exp(x-c)
    sum_exp = np.sum(exp)
    y = exp/sum_exp

    return y

a= np.array([0.3,2.9,4.])
print(softmax(a))