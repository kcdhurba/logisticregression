# imports 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV


# Generate dataset
X_tr, y_tr = make_moons(n_samples=300, noise=0.2, random_state=42)

# Plot
plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap='viridis', edgecolors='k')
plt.title("Non-linearly Separable Data (Moons)")
plt.show()


# sidmoid function
def sigmoid(z): 
    return (1/(1 + np.exp(-z)))

# layer function to for number of units 
def hidden_layer(X,Y, n_h:4): 
    """ 
    Args
    X (input size, number of examples (n,m)) -> input datasets
    y (output size, number of examples(1,m)) -> target lables

    Returns
    n_x --> size of the input layer 
    n_h --> size of the hidden layer
    n_y --> size of the output layer
    """

    n_x = X.shape[0] # number of features 
    n_y = Y.shape[0] # target variable
    return n_x, n_h, n_y

# parameters intialization 
def initialize_parameters(n_x, n_h, n_y): 
    """ 
    Args 
    n_x --> size of input layer 
    n_h --> size of hidden layer 
    n_y --> size of output layer

    Returns 
    parameters --> python dict containing parameters (W1, b1,W2, b2):  
    W1 -- weight matrix of hidden layer (n_h, n_x)
    b1 -- bias vector of hidden layer (n_h, 1)
    W2 -- weight matrix of output layer (n_y, n_h)
    b2 -- bias vector of output layer (n_y, 1) 
    """
    W1 = np.random.randn(n_h, n_x) * 0.01 # random initialization 
    b1 = np.zeros((n_h, 1)) 
    W2 = np.random.randn(n_y, n_h) * 0.01 
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1, 
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters
# forward propagation 
def forward_propagation(X, parameters): 
    """ 
    Args
    X -- Input data 
    parameters -- python dict containing initial values of parameters for each layers

    Returns:
    A2: sigmoid output of the second activation layer or final layer 
    cache: a dict containing Z1, A1, Z2, A2
    
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1 
    A1 = np.tanh(Z1) # hidden layer has tah(h) activation function. 
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2) # output layer has sigmoid activation function 

    assert(A2.shape == (1, X.shape[1])) # making sure output matches the target shape 
    # storing values for backpropagation
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache 

# computing cost 
def compute_cost(A2, Y): 
    """ 
    Args
    A2: sigmoid output of second activation of shape (1, number of examples)
    Y : True labels of shape (1, number of examples)
    Returns
    cost --> cross entropy cost 
    """
    m = Y.shape[1]
    cost = -(1/m)* np.sum(np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2)))

    cost = float(np.squeeze(cost)) # turns [[22]] into 22
    return cost 

# backpropagation 
def back_prop(parameters, cache, X, Y): 
    """ 
    backpropagation for 2 - layer NN
    Args: 
    parameters--> dict conatining W1, b1, W2, b2
    cache --> dict containing A2, and cache 
    X --> input data of shape (n_x, number of examples)
    Y --> true labels vector of shape(1, number of examples)

    Returns: 
    grads --> dict containing gradients dW1, db1, dW2, db2 
    
    """
    m = X.shape[1]
    W1 = parameters["W1"] # shape (n_h, n_x)
    W2 = parameters["W2"] # shape (n_y, n_h)
    A1 = cache["A1"] # shape(n_h, number of examples)
    A2 = cache['A2'] # shape(n_y, number of examples)

    # compute_gradients 
    dZ2 = A2 - Y # shape (n_y, number of examples)
    dW2 = (1/m) * np.dot(dZ2, A1.T) # shape (n_y, n_h)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True) # shape (n_y, 1)
    dZ1 = np.dot(W2.T, dZ2) * (1 - A1 * A1) # shape (n_h, n_x)
    dW1 = (1/m) * np.dot(dZ1, X.T) # shape (n_h, n_x)
    db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims=True)

    grads = {
        "dW1": dW1, 
        "db1": db1,
        "dW2": dW2, 
        "db2": db2
    }

    return grads

# update parameters 
def update_parameters(parameters, grads, learning_rate = 0.1): 
    """
    Args: 
    parameters --> dict cotaining parameters 
    grads --> gradient conatining gradient vector of each params 
    learning_rate --> learning rate for gradient descent

    Returns: 
    parameters --> updated dict with new values 

    """
    # update rule for each parameters ( in place update)
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]

    return parameters

# nn model to learn parameters 
def nn_model(X, Y, n_h, iterations = 1000, print_cost = False): 
    """ 
    Args: 
    X -- Traning dataset 
    Y -- target variable 
    n_h -- size of the hidden layer 
    iterations -- Number of iterations in gradient descent loop 
    
    Returns: 
    parameters -- parameters learnt by model. 
    """
    np.random.seed(2)
    n_x = hidden_layer(X, Y)[0]
    n_y = hidden_layer(X, Y)[2]
    
    # initialize parameters 
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(iterations): 
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = back_prop(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if print_cost and i%1000 == 0: 
            print(f'Cost at iter {i}: {cost}')

    return parameters
# train function 
def nn_train(nn_model): 
    """  
    Args
    nn_model: function arg
    Return
    parameters 
    """
    parameters = nn_model(X_tr.T, y_tr.T, 4, iterations = 1000, print_cost = True)

    return parameters

if __name__ == "__main__": 
    trained_parameters = nn_train(nn_model)

print(trained_parameters)













