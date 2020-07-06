import numpy as np

def identity_function(x):
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def init_network():
    nn = {}
    nn['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    nn['b1'] = np.array([0.1, 0.2, 0.3])
    nn['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    nn['b2'] = np.array([0.1,0.2])
    nn['W3'] = np.array([[0.1,0.3],[0.2, 0.4]])
    nn['b3'] = np.array([0.1, 0.2])

    return nn

def forward(nn,X):
    W1,W2,W3 = nn['W1'],nn['W2'],nn['W3']
    b1,b2,b3 = nn['b1'],nn['b2'],nn['b3']

    A1 = np.dot(X,W1)+b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1,W2)+b2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2,W3)+b3
    y = identity_function(A3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network,x)
print(y)
#출력층의 활성화 함수인 identity_function을 시그마라한다.
