"""
This project aims to create the steps and foundation of a Deep Neural Network
Steps are as follow: 
1- Initialize parameters (W, b)
2- Forward propagation
3- Compute cost
4- Backward propagation
5- Update parameters
6- Predit
"""

import numpy as np

# Specify each layer's dimention (units)
layer_dims = []

# X is the input vector for m examples and Y is the corresponding class
X = []
Y = []

def initialize_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        W = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        b = np.zeros((layer_dims[l], 1))

        parameters['W' + str(l)] = W
        parameters['b' + str(l)] = b
    return parameters

def relu(x):
    return max(0.0, x)

def sigmoid(x):
    return 1 / 1 + (np.exp(-x))

def forward_propagation(X, parameters):
    L = len(layer_dims)
    A = X
    caches = {}
    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A = relu(Z)

        caches['Z' + str(l)] = Z
        caches['A' + str(l)] = A
    
    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = sigmoid(ZL)

    caches['Z' + str(L)] = ZL
    caches['A' + str(L)] = AL

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = - (np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T)) / m
    cost = np.squeeze(cost)
    return cost

def sigmoid_derivitive(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu_derivative(x):
    if x < 0:
        return 0
    elif x >= 0:
        return 1

def backward_propagation(AL, Y, caches, parameters):
    L = len(caches) / 2
    grads = {}
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))

    # Backward for the last layer (sigmoid function)
    ZL = caches['Z' + str(L)]
    A_prev = caches['A' + str(L - 1)]
    W = parameters['W' + str(L)]

    dZ = np.multiply(dAL, sigmoid_derivitive(ZL))
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    grads['dA' + str(L-1)] = dA_prev
    grads['dW' + str(L)] = dW
    grads['db' + str(L)] = db

    for l in reversed(range(1, L)):
        Z = caches['Z' + str(l)]
        A_prev = caches['A' + str(l - 1)]
        dA = grads['dA' + str(l)]
        W = parameters['W' + str(l)]

        dZ = np.multiply(dA, relu_derivative(Z))
        dW = np.dot(dZ, A_prev.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        grads['dA' + str(l - 1)] = dA_prev
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) / 2
    # What is the reason for the copy?
    parameters = copy.deepcopy(parameters)

    for l in range(1, L+1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - grads['dW' + str(l)] * learning_rate
        parameters['b' + str(l)] = parameters['b' + str(l)] - grads['db' + str(l)] * learning_rate

    return parameters


