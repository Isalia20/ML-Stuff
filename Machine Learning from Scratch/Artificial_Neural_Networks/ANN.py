# WIP
import numpy as np


class FeedForwardNeuralNetwork:

    def __init__(self,
                 random_state=42,
                 output_threshold=0.5,
                 x=None,
                 learning_rate = 0.01
                 ):
        if x is None:
            raise Exception("You need to initialize the class with X")
        self.random_state = random_state
        self.weight_matrices = {}
        self.output_threshold = output_threshold
        self.cache = {}
        self.x = x
        self.learning_rate = learning_rate
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

    @staticmethod
    def sigmoid_function(z):
        exp_z = np.array([np.math.exp(i) for i in z]).reshape(z.shape[0], z.shape[1])
        return 1/(1 + exp_z)

    def feed_forward(self, x):
        layers = len(self.network_architecture)
        x_input = x
        for layer_index in range(1, layers):
            weight_matrix = self.weight_matrices[layer_index][0]
            bias = self.weight_matrices[layer_index][1]
            if self.network_architecture[layer_index][2]:
                pre_activation = x_input @ weight_matrix + bias
            else:
                pre_activation = x_input @ weight_matrix

            if self.network_architecture[layer_index][1] == "relu":
                activation = pre_activation * (pre_activation > 0)
            elif self.network_architecture[layer_index][1] == "logistic":
                activation = self.sigmoid_function(pre_activation)
            self.cache[layer_index] = (x_input,pre_activation, activation)
            x_input = activation
        return x_input

    def make_prediction(self, output):
        y_pred = (output >= self.output_threshold) * 1
        return y_pred

    @staticmethod
    def calculate_loss(y, y_pred):
        y = np.reshape(y, (-1,))
        y_pred = np.reshape(y_pred, (-1,))
        loss = -np.sum(y * np.array([np.math.log(i) for i in y_pred]) + (1 - y) * np.array([np.math.log(1 - i) for i in y_pred]))
        return loss

    def backwards_prop(self, y):
        m = self.x.shape[0]
        y = y.reshape((-1, 1))
        last_layer = list(self.cache.keys())[-1]
        a = self.cache[last_layer][2]
        z = self.cache[last_layer][1]
        x = self.cache[last_layer][0]
        dL_da = - (y/a + (1-y)/(1-a))
        da_dz = self.sigmoid_function(z) * (1 - self.sigmoid_function(z))
        dz_dw = x
        dz_db = 1
        dL_dw = (1/m * np.sum(dL_da * da_dz * dz_dw, axis=0)).reshape((-1,1))
        dL_db = (1/m * np.sum(dL_da * da_dz * dz_db, axis=0)).reshape((-1,1))
        w = self.weight_matrices[last_layer][0]
        b = self.weight_matrices[last_layer][1]
        w = w - self.learning_rate * dL_dw
        b = b - self.learning_rate * dL_db
        return w,b
