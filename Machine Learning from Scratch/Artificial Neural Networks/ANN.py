# WIP

import numpy as np

class NeuralNetwork:

    def __init__(self,
                 n_layers = 1,
                 activation = "relu",
                 solver = "adam",
                 alpha = 0.01,
                 random_state = 42):
        self.n_layers = n_layers
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.random_state = random_state

    def 