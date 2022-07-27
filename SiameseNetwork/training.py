# WIP
# TODO take a look at architecture and improve it to being able to predict with 1 tensor
# TODO add functions for getting ready prediction texts with single function
import os
os.chdir("SiameseNetwork_Final")
import sys
sys.path[-1] = os.getcwd()
from keras import layers
import tensorflow as tf
import keras
from DataPreProcessing import PreProcessData
from TripletLoss import triplet_loss
from EmbeddingLayers import EmbeddingLayer
from DataPreProcessing import data_generator
import numpy as np

# PreProcessing the dataframe
preprocess = PreProcessData()
triplets = preprocess.create_triplets("train_quora.csv", 100000)

# Recovering anchors and positives from triplets
anchors = [i[0] for i in triplets]
positives = [i[1] for i in triplets]

# Converting to tensors
anchors = tf.convert_to_tensor(anchors)
positives = tf.convert_to_tensor(positives)

# Reshaping for model inputs
anchors = tf.reshape(anchors, (anchors.shape[0], 1, anchors.shape[1]))
positives = tf.reshape(positives, (positives.shape[0], 1, positives.shape[1]))

# Maximum length of a question
max_len = anchors.shape[2]
# Freeing up space
del triplets

# Setting up a model
embed_layer = EmbeddingLayer()
vocab_size = 10000
d_model = 512
units = 100
batch_size = 32
model = embed_layer.build_model(vocab_size=vocab_size, d_model=d_model, units=units, input_len=max_len,
                                batch_size=batch_size)

# Setting up an input for siamese model
anchor_input = layers.Input(name="anchors", shape=(1, max_len), batch_size=batch_size)
positive_input = layers.Input(name="positive", shape=(1, max_len), batch_size=batch_size)
embedding_anchor = model(anchor_input)
embedding_positive = model(positive_input)

# Setting up output for siamese model
output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive], axis=1)

# Setting up a model
net = tf.keras.models.Model([anchor_input, positive_input], output)
# Summary
net.summary()

# Initializing data generator for training
data_generator = data_generator(batch_size_input=batch_size, embedding_size=units,
                                    anchors_input=anchors, positives_input=positives)

# Compiling and training the model after specifying parameters
epochs = 10
steps_per_epoch = int(anchors.shape[0] / batch_size)
margin = 0.25
net.compile(loss=triplet_loss(units=units, batch_size=batch_size, margin=margin),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
net.fit(data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

# Saving weights
net.save_weights('model/siamese_model.h5')

# model.load_weights("model/siamese_model.h5")
