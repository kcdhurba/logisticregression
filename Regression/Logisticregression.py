import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, Weights, bias, X_train, Y_train, hyperparameter, learning_rate, iterations):
        self.Weights = Weights  # weight parameter (1, n)
        self.bias = bias  # scalar
        self.X_train = X_train  # training data of size (n, m) => n features, m examples
        self.Y_train = Y_train  # target variables (1, m)
        self.learning_rate = learning_rate  # scalar
        self.hyperparameter = hyperparameter  # scalar (regularization strength)
        self.iterations = iterations  # number of iterations

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def prediction(self):
        """Make predictions using the learned weights and bias"""
        Z = np.dot(self.Weights, self.X_train) + self.bias  # linear combination
        A = self.sigmoid(Z)  # applying sigmoid to the linear combination
        return A  # predicted probabilities (1, m)

    def compute_cost(self):
        """Compute the cost function (logistic loss with L2 regularization)"""
        m = self.X_train.shape[1]  # number of examples
        Y_pred = self.prediction()
        
        # Binary cross-entropy loss
        cost = -1/m * np.sum(np.multiply(self.Y_train, np.log(Y_pred)) + np.multiply(1 - self.Y_train, np.log(1 - Y_pred)))
        
        # Add L2 regularization term
        cost += (self.hyperparameter / (2 * m)) * np.sum(np.square(self.Weights))  # L2 regularization
        
        return cost

    def compute_gradient(self):
        """Compute gradients of cost function with respect to Weights and bias"""
        m = self.X_train.shape[1]  # number of examples
        A = self.prediction()  # predicted probabilities
        dz = A - self.Y_train  # gradient of the loss with respect to predictions
        
        db = np.sum(dz) / m  # gradient with respect to bias (scalar)
        dw = np.dot(dz, self.X_train.T) / m  # gradient with respect to weights (1, n)
        
        # Regularization term for dw (L2 regularization)
        dw += (self.hyperparameter / m) * self.Weights  # L2 regularization for weights
        
        return dw, db

    def gradient_descent(self):
        """Optimize Weights and bias using gradient descent"""
        cost_history = []
        w = self.Weights
        b = self.bias

        for i in range(self.iterations):
            # Compute gradients
            dw, db = self.compute_gradient()

            # Update weights and bias using the gradients and learning rate
            self.Weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Record the cost every 100 iterations
            if i % 100 == 0:
                cost = self.compute_cost()
                cost_history.append(cost)
                print(f'Cost at iteration {i}: {cost}')

        return self.Weights, self.bias, cost_history

# Example usage:
# Define hyperparameters
weights = np.random.randn(1, 10)  # Initial weights (1, n) where n=10
bias = 0.0  # Initial bias
X_train = np.random.randn(10, 100)  # Training data (n, m) where n=10 features, m=100 examples
Y_train = np.random.randint(0, 2, (1, 100))  # Binary target (1, m)

learning_rate = 0.01
iterations = 1000
hyperparameter = 0.1  # Regularization strength

# Initialize logistic regression model
log_reg = LogisticRegression(weights, bias, X_train, Y_train, hyperparameter, learning_rate, iterations)

# Train the model using gradient descent
weights, bias, cost_history = log_reg.gradient_descent()

# Plot cost over iterations
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function during Gradient Descent')
plt.show()
