import numpy as np
from Artificial_Neural_Networks.ANN import FeedForwardNeuralNetwork
from sklearn import datasets

import numpy as np

# training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
#
# data = datasets.load_breast_cancer()
# X, y = data.data, data.target



ffn = FeedForwardNeuralNetwork(x=X)
ffn.add_change_layer(1, 5, "relu", bias_neuron= True)
ffn.add_change_layer(2, 1, "logistic", bias_neuron=True)
ffn.generate_weight_matrices()

ffn.cache_backprop[2]

ffn.train(X, y, epochs = 100)



ffn.feed_forward(X)
ffn.backwards_prop(y)