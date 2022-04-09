import numpy as np
from Artificial_Neural_Networks.ANN import FeedForwardNeuralNetwork

X = np.random.normal(size = (100,5))
y = np.random.randint(0,2,size = 100)

ffn = FeedForwardNeuralNetwork(x = X)

ffn.add_change_layer(1,3,"relu",bias_neuron=True)
ffn.add_change_layer(2,1,"logistic",bias_neuron=True)
ffn.generate_weight_matrices()

y_pred = ffn.feed_forward(X)

ffn.calculate_loss(y,y_pred)

y_pred