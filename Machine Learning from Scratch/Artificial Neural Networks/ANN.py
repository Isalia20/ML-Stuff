# WIP
import numpy as np


class FeedForwardNeuralNetwork:

    def __init__(self,
                 random_state=42,
                 ):
        self.random_state = random_state
        self.network_architecture = {}
        self.weight_matrices = {}

    def _initialize_network(self, x):
        self.network_architecture[0] = x.shape[1]

    def add_change_layer(self, layer_index, n_neurons, activation_function, bias_neuron = True):
        if layer_index < 0:
            raise Exception("You can't change/add layer before input")
        elif layer_index == 0:
            raise Exception("You can't change input layer's neuron count. Change shape of X and reinitialize class")
        self.network_architecture[layer_index] = (n_neurons, activation_function, bias_neuron)

    def _generate_weight_matrices(self, activation_function):
        for layer_index in range(1, self.network_architecture.keys()):
            self.weight_matrices[layer_index] = \
                np.random.normal(0, 1, (self.network_architecture[layer_index - 1],
                                        self.network_architecture[layer_index]))

    def _feed_forward(self, last_layer_output, layer_index):
        x = np.matrix(last_layer_output)
        weight_matrix = self.weight_matrices[layer_index]
        if self.network_architecture[layer_index][2]:
            pre_activation = np.matmul(x, weight_matrix)

        pre_activation = np.matmul(x, weight_matrix)
        if self.network_architecture[layer_index][1] == "relu":
            activation = pre_activation * (pre_activation > 0)
        # Other activations need to be implemented here as well
        return activation

    def _backwards_prop(self, w):
        print("tmp")









