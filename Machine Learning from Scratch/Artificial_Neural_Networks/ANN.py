# WIP
import numpy as np


class FeedForwardNeuralNetwork:

    def __init__(self,
                 random_state=42,
                 output_threshold=0.5,
                 x=None
                 ):
        if x is None:
            raise Exception("You need to initialize the class with X")
        self.random_state = random_state
        self.weight_matrices = {}
        self.output_threshold = output_threshold
        self.x = x
        # Initializing network architecture with input shape n_neurons, activation_function(None at layer 0)
        # and bias neuron(also False) boolean
        self.network_architecture = {0: (self.x.shape[1], None, False)}

    def add_change_layer(self, layer_index, n_neurons, activation_function, bias_neuron = True):
        if layer_index < 0:
            raise Exception("You can't change/add layer before input (negative layer_index)")
        elif layer_index == 0:
            raise Exception("You can't change input layer's neuron count. Change shape of X and reinitialize class")
        self.network_architecture[layer_index] = (n_neurons, activation_function, bias_neuron)

    def generate_weight_matrices(self):
        for layer_index in range(1, len(self.network_architecture.keys())):
            self.weight_matrices[layer_index] = \
                (np.random.normal(0, 1, (self.network_architecture[layer_index - 1][0],
                                        self.network_architecture[layer_index][0])),  # Normal weights
                 np.random.normal(0, 1, (1, self.network_architecture[layer_index][0])))  # Bias

    def feed_forward(self, previous_layer_output, layer_index):
        x = previous_layer_output
        weight_matrix = self.weight_matrices[layer_index][0]
        bias = self.weight_matrices[layer_index][1]
        if self.network_architecture[layer_index][2]:
            pre_activation = x @ weight_matrix + bias
        else:
            pre_activation = x @ weight_matrix

        if self.network_architecture[layer_index][1] == "relu":
            activation = pre_activation * (pre_activation > 0)
        elif self.network_architecture[layer_index][1] == "logistic":
            activation = 1 / (1 + np.exp(-pre_activation))

        return activation

    def make_prediction(self, last_output):
        y_pred = last_output >= self.output_threshold
        return y_pred

    @staticmethod
    def _calculate_loss(y, y_pred):
        loss = - (y * np.math.log(y_pred) + (1 - y) * np.math.log(1 - y_pred))
        return loss

    def _backwards_prop(self, y, y_pred):
        return 2
