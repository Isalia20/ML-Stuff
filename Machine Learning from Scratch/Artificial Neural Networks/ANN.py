#WIP
import numpy as np

np.array([4,5,3])


class NeuralNetwork:

    def __init__(self,
                 input_shape=(200,5),
                 random_state=42):
        self.input_shape = input_shape

    def _initialize_network(self):
        m = self.input_shape[0]
        n = self.input_shape[1]
