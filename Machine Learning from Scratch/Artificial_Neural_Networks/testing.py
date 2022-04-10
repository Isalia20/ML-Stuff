import numpy as np
from Artificial_Neural_Networks.ANN import FeedForwardNeuralNetwork

X = np.random.normal(size = (100,5))
y = np.random.randint(0,2,size = 100)

ffn = FeedForwardNeuralNetwork(x = X)

ffn.add_change_layer(1,3,"relu",bias_neuron=True)
ffn.add_change_layer(2,1,"logistic",bias_neuron=True)
#ffn.add_change_layer(3,1,"logistic", bias_neuron=True)
ffn.generate_weight_matrices()


ffn.train(X,y,10)



y_pred = ffn.feed_forward(X)

z = np.ones((100,100))

z.shape


np.exp(z)

np.array([np.math.exp(i) for i in z]).reshape((z.shape[0], z.shape[1]))



np.array([np.math.exp(i) for i in z])






ffn.backwards_prop(y)



a.shape


ffn.weight_matrices[]



a.shape

dL_dw.shape
dL_db.shape

ffn.weight_matrices[2][0].shape

dL_dw.shape
dL_db.shape



dL_dw.shape
dL_db.shape

ffn.weight_matrices[1][1].shape


dL_da.shape
da_dz.shape
x.shape


da_dz.shape
dz_dw.shape


dL_da.shape
da_dz.shape
dz_da.shape


dL_da.shape
da_dz.shape
dz_dw.shape

1/100 * np.sum(dL_dz,axis = 0, keepdims= True)


a.shape


np.dot(dL_dz.T,x).shape


np.dot((dL_da * da_dz)


(dL_da * da_dz).shape





da_dz.shape
dz_dw.shape





tmp = np.dot(dL_da.T, da_dz)

np.dot(dz_dw,tmp)



np.dot(dz_dw.shape)
