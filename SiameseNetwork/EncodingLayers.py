import tensorflow as tf
from keras import layers
import keras


class MeanNormalizationLayer(keras.layers.Layer):
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, mask=None, training=None, initial_state=None):
        return inputs / tf.math.sqrt(tf.math.reduce_sum(inputs * inputs, axis=-1, keepdims=True))


class EmbeddingLayer:
    def __init__(self):
        self.MeanNormalizeLayer = MeanNormalizationLayer

    def build_model(self, vocab_size, d_model, units, input_len, batch_size):
        model = keras.Sequential(
            [
                layers.Embedding(input_dim=vocab_size,
                                 output_dim=d_model,
                                 input_length=input_len,
                                 input_shape=(1, input_len),
                                 batch_size=batch_size),
                layers.Reshape((input_len, d_model), input_shape=(1, input_len, d_model), batch_size=batch_size),
                layers.LSTM(units=units, input_dim=(batch_size, input_len, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model)),
                self.MeanNormalizeLayer()
            ]
        )
        return model
