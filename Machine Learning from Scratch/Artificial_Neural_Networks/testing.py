import numpy as np
from Artificial_Neural_Networks.ANN import FeedForwardNeuralNetwork

X = np.random.normal(size = (100,5))
y = np.random.randint(0,2,size = 100)

ffn = FeedForwardNeuralNetwork(x = X)

ffn.add_change_layer(1,3,"relu",bias_neuron=True)
ffn.add_change_layer(2,1,"logistic",bias_neuron=True)
ffn.generate_weight_matrices()

y_pred = ffn.feed_forward(X)

dL_da, da_dz, dz_da_w = ffn.backwards_prop(y)

dL_da.shape
da_dz.shape
dz_da_w.shape



dL_da.shape
da_dz.shape
dz_dw.shape





dL_dw_last.shape
dw_da.shape
da_dz.shape
dz_dw.shape



ffn.network_architecture




tmp = ffn.weight_matrices[1]

tmp = (2,1)


for index, layer_index in enumerate(reversed(range(1, layers))):
    print(index,layer_index)


ffn.weight_matrices[2][0]


layers = len(ffn.network_architecture)

layers




dL_dw,dL_db,w,b = ffn.backwards_prop(y)

for i in reversed(range(1,6)):
    print(i)


ffn.network_architecture



(X > 0).astype(int)






