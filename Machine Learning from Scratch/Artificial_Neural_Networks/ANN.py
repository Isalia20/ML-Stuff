# WIP
import numpy as np


class FeedForwardNeuralNetwork:

    def __init__(self,
                 random_state=42,
                 output_threshold=0.5,
                 x=None,
                 learning_rate=0.1
                 ):
        if x is None:
            raise Exception("You need to initialize the class with X")
        self.random_state = random_state
        self.weight_matrices = {}
        self.output_threshold = output_threshold
        self.cache = {}
        self.cache_backprop = {}
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
                (np.random.normal(0, 0.1, (self.network_architecture[layer_index - 1][0],
                                        self.network_architecture[layer_index][0])))  # Weights

    @staticmethod
    def sigmoid_function(z):
        exp_z = np.exp(-z)
        return 1/(1 + exp_z)

    def feed_forward(self, x):
        layers = len(self.network_architecture)
        x_input = x
        for layer_index in range(1, layers):
            weight_matrix = self.weight_matrices[layer_index]
            if self.network_architecture[layer_index][2]:
                pre_activation = (x_input @ weight_matrix)
            else:
                pre_activation = x_input @ weight_matrix

            if self.network_architecture[layer_index][1] == "relu":
                activation = pre_activation * (pre_activation > 0)
            elif self.network_architecture[layer_index][1] == "logistic":
                activation = self.sigmoid_function(pre_activation)
            self.cache[layer_index] = (x_input, pre_activation, activation)
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
        layers = len(self.network_architecture)
        for index, layer_index in enumerate(reversed(range(1, layers))):
            a_backprop = self.cache[layer_index][2]
            z_backprop = self.cache[layer_index][1]
            x_backprop = self.cache[layer_index][0]
            if index == 0:
                dL_da = -(1/m) * (y / a_backprop + (1-y) / (1-a_backprop))  # error here
                if self.network_architecture[layer_index][1] == "logistic":
                    da_dz = self.sigmoid_function(z_backprop) * (1 - self.sigmoid_function(z_backprop))  # Need to implement other functions as well
                dL_dz = dL_da * da_dz
                dL_dw = np.dot(x_backprop.T, dL_dz)
                self.cache_backprop[layer_index] = (dL_dw)
            else:
                dL_dz = np.sum(dL_dz)
                dz_da = self.weight_matrices[layer_index + 1]
                if self.network_architecture[layer_index][1] == "relu":
                    da_dz = (z_backprop > 0).astype(int)
                elif self.network_architecture[layer_index][1] == "logistic":
                    da_dz = self.sigmoid_function(z_backprop) * (1 - self.sigmoid_function(z_backprop))
                dz_dw = x_backprop
                tmp = (dz_dw.T @ da_dz)
                tmp = tmp * dz_da.T
                tmp = tmp * dL_dz
                dL_dw = tmp
                # dL_dw = (1/m) * np.matmul(dz_dw.T, dL_dz)
                self.cache_backprop[layer_index] = (dL_dw)

        for layer_index in reversed(range(1, layers)):
            w = self.weight_matrices[layer_index]
            # b = self.weight_matrices[layer_index][1]
            #print(self.cache_backprop[layer_index])
            w = w - self.learning_rate * self.cache_backprop[layer_index]
            # b = b - self.learning_rate * self.cache_backprop[layer_index][1]
            self.weight_matrices[layer_index] = (w)

    def train(self, x, y, epochs):
        for i in range(epochs):
            y_pred = self.feed_forward(x)
            print("Loss for epoch " + str(i) + " is " + str(self.calculate_loss(y, y_pred)))
            self.backwards_prop(y)

# Not finished