# -*- coding: utf-8 -*-
import cPickle
import gzip
import numpy as np
import random

class MiniNN(object):

    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        # Generate random biases for all nodes not in the input layer
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        # Generate random weights for all edges
        self.weights = [np.random.randn(y, x) for x, y in zip(
            layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, a):
        # Return the output of the network given an input a
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, num_epochs, mini_batch_size, eta, test_data=None):
        if test_data: num_tests = len(test_data)

        num_train = len(training_data)

        for j in xrange(num_epochs):
            # Randomly shuffle the training data and split into mini batches
            random.shuffle(training_data)

            mini_batches = [training_data[k:k + mini_batch_size]
                for k in range(0, num_train, mini_batch_size)]

            # Train the network for each mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print('---')
                print("Epoch {0}: {1} / {2}".format(j,
                      self.evaluate(test_data), num_tests))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # Calculate the gradient of the cost function via back propagation
            delta_nabla_b, delta_nabla_w = self.back_prop(x, y)

            # Sum up all the changes
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Average the changes over the mini batch and update the weights and biases
        self.weights = [w - (eta / len(mini_batch)) *
                             nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b,
                            nb in zip(self.biases, nabla_b)]

    def back_prop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward pass
        activation = x
        activations = [x]
        zs = [0]

        y = np.asarray(y)

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        (x, y) = test_results[0]
        return sum(int(x == y) for (x,y) in test_results)

    def cost_derivative(self, output, y):
        return (output - y)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def reLU(z):
    if (z > 0.0):
        return z
    else:
        return 0.0

def translate_output(o):
    t = np.zeros((10,1))
    t[o] = 1.0
    return t

if __name__ == "__main__":
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    tr_d, va_d, te_d = cPickle.load(f)
    f.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [translate_output(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    print('Finished loading in data.')

    network = MiniNN([784, 30, 10])
    network.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data)
