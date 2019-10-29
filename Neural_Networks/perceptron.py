

import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=500):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

# ------- Generating the dataset using make_blobs -------
X,Y = make_blobs(n_samples=800, centers=2, n_features=2, random_state=2)
plt.style.use("seaborn")
plt.scatter(X[:,0],X[:,1],c=Y,cmap = plt.cm.Accent)
plt.show()

# -------- Splitting train and test --------- 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.3,random_state = 101)

# -------- Predicting using Perceptron class --------
p = Perceptron()
p.fit(Xtrain, Ytrain)
pred = p.predict(Xtest)

print(p.accuracy(Ytest,pred))