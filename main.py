from NeuralNetwork import *

X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
# y = np.array([[0], [1], [1], [0]], dtype=float) # xor
y = np.array(([0], [0], [0], [1]), dtype=float)  # and

network = NeuralNetwork(learning_rate=0.1)
iterations = 5000

if __name__ == '__main__':
    for i in range(iterations):
        network.train(X, y)

        ten = iterations // 10
        if i % ten == 0:
            print(f'Epoch {i} MSE: {np.mean(np.square(y - network.output))}')

    for i in range(len(X)):
        print(f'Prediction for {X[i]}: {network.predict(X[i])} real target: {y[i]}')
