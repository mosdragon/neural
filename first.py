import numpy as np
# import theano
import sys

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return (np.exp(-z)) / ((1 + np.exp(-z)) ** 2)

def mean_squared_error(guesses, actuals):
    errors = [(guess - actual) ** 2 for guess, actual in zip(guesses, actuals)]
    return np.mean(errors)

class NeuralNet(object):
    def __init__(self, n_input, n_hidden, n_output=1,
        activation=sigmoid):
#         Define hyperparameters
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.W1 = np.random.randn(n_input, n_hidden)
        self.W2 = np.random.randn(n_hidden, n_output)

        self.activation = activation


    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.activation(self.z3)

        return yHat


    def cost_function_prime(self, X, y):
        self.yHat = self.forward(X)

        delta3 = np.multiply(- (y - self.yHat), sigmoid_prime(self.z3))
        djdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * sigmoid_prime(self.z2)
        djdW1 = np.dot(X.T, delta2)

        return djdW1, djdW2

    def train(self, X, y, epochs=10, learning_rate=0.3):
        # learning_rate = 0.01
        for _ in range(epochs):
            djdW1, djdW2 = self.cost_function_prime(X, y)
            # for X, y in zip(attributes, targets):
            #     djdW1, djdW2 = self.cost_function_prime(X, y)
            self.W1 -= djdW1 * learning_rate
            self.W2 -= djdW2 * learning_rate



def main():
    # X  = np.random.rand(30, 2)
    # y = np.random.rand(30, 1)

    # X = (hours sleeping, hours studying), y = Score on test
    X = np.array(([3,5], [5,1], [10,2], [3, 2]), dtype=float)
    y = np.array(([75], [82], [93], [22]), dtype=float)

    # Normalize
    X = X/np.amax(X, axis=0)
    y = y/100 #Max test score is 100

    # print "X: {}".format(X)
    # print "y: {}".format(y)

    # print "X: {}".format(X)
    # print "y: {}".format(y)

    # X_train, y_train = X[:21], y[:21]
    # X_test, y_test = X[21:], y[21:]

    # print "X: {}".format(X_train)
    # print "y: {}".format(y_train)

    net = NeuralNet(2, 3, 1)

    yHats = [net.forward(data) for data in X]
    before_error = mean_squared_error(yHats, y)
    print "Before Training: {}".format(before_error)
    print "y: {}".format(y)
    print "yHat: {}".format(yHats)

    djdw1, djdw2 = net.cost_function_prime(X, y)
    # print djdw1, djdw2
    net.train(X, y, 10000)

    yHats = [net.forward(data) for data in X]
    after_error = mean_squared_error(yHats, y)
    print "After Training: {}".format(after_error)

    print "y: {}".format(y)
    print "yHat: {}".format(yHats)


if __name__ == '__main__':
    main()
