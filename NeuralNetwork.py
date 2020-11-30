import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate=0.2):
        self.learning_rate = learning_rate
        self.output = None
        # weights=random values
        self.weights = [
            np.random.uniform(low=-0.2, high=0.2, size=(2, 2)),  # input layer
            np.random.uniform(low=-2, high=2, size=(2, 1))  # hidden layer
        ]

    def activate(self, y):
        # return self.sigmoid(y)
        return self.tangent(y)

    def activate_derivative(self, y):
        # return self.sigmoid_derivative(y)
        return self.tangent_derivative(y)

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def sigmoid_derivative(self, y):
        return y * (1 - y)

    def tangent(self, y):
        return np.tanh(y)

    def tangent_derivative(self, y):
        return 1.0 - y ** 2

    def forward(self, x_values):
        input_layer = x_values
        hidden_layer = self.activate(np.dot(input_layer, self.weights[0]))
        output_layer = self.activate(np.dot(hidden_layer, self.weights[1]))

        self.layers = [
            input_layer,
            hidden_layer,
            output_layer
        ]

        return output_layer

    def backward(self, target_output, actual_output):
        err = target_output - actual_output

        for backward in range(2, 0, -1):
            gradient = err * self.activate_derivative(self.layers[backward])
            self.weights[backward - 1] += self.learning_rate * np.dot(self.layers[backward - 1].T, gradient)
            err = np.dot(gradient, self.weights[backward - 1].T)

    def train(self, x_values, target):
        self.output = self.forward(x_values)
        self.backward(target, self.output)

    def predict(self, x_values):
        return self.forward(x_values)
