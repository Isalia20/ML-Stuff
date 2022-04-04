# WIP
import numpy as np


class FeedForwardNeuralNetwork:

    def __init__(self,
                 random_state=42):
        self.random_state = random_state
        self.network_architecture = {}
        self.weight_matrices = {}

    def _initialize_network(self, x):
        self.network_architecture[0] = x.shape[1]

    def add_change_layer(self, layer_index, n_neurons):
        if layer_index < 0:
            raise Exception("You can't change/add layer before input")
        elif layer_index == 0:
            raise Exception("You can't change input layer's neuron count. Change shape of X and reinitialize class")
        self.network_architecture[layer_index] = n_neurons

    def _generate_weight_matrices(self):
        for layer_index in range(1, self.network_architecture.keys()):
            self.weight_matrices[layer_index] = \
                self.network_architecture = \
                np.random.normal(0, 1,
                                 (self.network_architecture[layer_index - 1],
                                  self.network_architecture[layer_index]))

    # def _feed_forward(self, x):
    #     x = np.matrix(x)
    #     np.matmul(x, self.weight_matrices[0])