# WIP
# TODO add removing stopwords as they dont add much meaning in differentiating sentences from each other
# TODO take a look at architecture and improve it to being able to predict with 1 tensor
# TODO add functions for getting ready prediction texts with single function
# TODO look at triplet loss once again
# TODO finish data generator function
import pandas as pd
import numpy as np
import nltk
from keras import layers
import tensorflow as tf
import keras
from nltk.stem import PorterStemmer
import string


class PreProcessData:
    def __init__(self):
        # Vocabulary dictionary where key is word and values are numbers
        self.vocab_dict = None
        # Vocabulary dictionary where key is integer and values are words
        self.vocab_dict_integer = None
        self.max_len = None

    @staticmethod
    def _process_dataframe(dataframe_path):
        dataframe = pd.read_csv(dataframe_path)
        dataframe = dataframe[dataframe["is_duplicate"] == 1]
        Q1 = dataframe["question1"].to_numpy()
        Q2 = dataframe["question2"].to_numpy()
        is_duplicate = dataframe["is_duplicate"].to_numpy()

        corpus = (Q1, Q2, is_duplicate)
        return corpus

    def _preprocess_data(self, dataframe_path, n_rows=5000):
        corpus = self._process_dataframe(dataframe_path)
        ps = PorterStemmer()
        Q1 = corpus[0][:n_rows]
        Q2 = corpus[1][:n_rows]
        is_duplicate = corpus[2][:n_rows]
        Q1_processed = []
        Q2_processed = []
        duplicates = []
        for i in range(len(is_duplicate)):
            # initializing lists for appending processed sentences
            sentence_1 = Q1[i]
            sentence_2 = Q2[i]
            duplicate = is_duplicate[i]
            Q1_sentence = []
            Q2_sentence = []
            try:
                sentence_1_tokenized = nltk.word_tokenize(sentence_1)
                sentence_2_tokenized = nltk.word_tokenize(sentence_2)
            except TypeError:
                continue
            for word_1, word_2 in zip(sentence_1_tokenized, sentence_2_tokenized):
                word_processed_1 = ps.stem(word_1.lower())
                word_processed_1 = word_processed_1.translate(str.maketrans('', '', string.punctuation))
                Q1_sentence.append(word_processed_1)

                word_processed_2 = ps.stem(word_2.lower())
                word_processed_2 = word_processed_2.translate(str.maketrans('', '', string.punctuation))
                Q2_sentence.append(word_processed_2)

            Q1_processed.append(Q1_sentence)
            Q2_processed.append(Q2_sentence)
            duplicates.append(duplicate)

        return Q1_processed, Q2_processed, is_duplicate

    def get_max_sentence_len(self, Q1_processed, Q2_processed):
        Q_processed = Q1_processed + Q2_processed
        max_len = 0
        for sentence in Q_processed:
            sentence_len = len(sentence)
            if sentence_len > max_len:
                max_len = sentence_len
        self.max_len = max_len

    def pad_sentence(self, sentence, pad):
        sentence_len = len(sentence)
        max_len = 2 ** (int(np.log2(self.max_len)) + 1)  # For better computations on GPU
        sentence = sentence + [pad] * (max_len - sentence_len)
        return sentence

    @staticmethod
    def _create_word_indices(Q1_processed, Q2_processed):
        vocab_q1 = set()
        vocab_q2 = set()
        questions = [Q1_processed, Q2_processed]

        for q_index in range(len(questions)):
            for sentence in questions[q_index]:
                for word in sentence:
                    if q_index == 0:
                        vocab_q1.add(word)
                    else:
                        vocab_q2.add(word)

        vocab_dict = {}

        vocab_sets = [vocab_q1, vocab_q2]

        for vocab_set in vocab_sets:
            # We start from 1 due to reason of adding padding tokens equal 0
            for number, items in enumerate(vocab_set, 1):
                word = items
                vocab_dict[word] = number

        return vocab_dict

    def _prepare_train_data(self, dataframe_path, n_rows):
        Q1_processed, Q2_processed, is_duplicate = self._preprocess_data(dataframe_path, n_rows)
        self.get_max_sentence_len(Q1_processed, Q2_processed)
        self.vocab_dict = self._create_word_indices(Q1_processed, Q2_processed)
        y = is_duplicate  # x is defined later
        Q1_integers = []
        Q2_integers = []
        questions = [Q1_processed, Q2_processed]
        questions_integers = [Q1_integers, Q2_integers]
        for question_index in range(len(questions)):
            for sentence in questions[question_index]:
                word_nums = []
                for word in sentence:
                    word_num = self.vocab_dict.get(word)
                    word_nums.append(word_num)
                # (sentence, pad, max_len)
                word_nums = self.pad_sentence(word_nums, 0)
                question_integers_list = questions_integers[question_index]
                question_integers_list.append(word_nums)
        x = questions_integers
        return x, y

    def preprocess_prediction_sentence(self, sentence, pad):
        """
        Function for preprocessing prediction sentence after model is fitted
        :return:
        Sentence preprocessed, final return is a list with integers
        """
        ps = PorterStemmer()
        sentence = nltk.word_tokenize(sentence)
        new_sentence = []
        for word in sentence:
            word = ps.stem(word.lower())
            word = word.translate(str.maketrans('', '', string.punctuation))
            word_integer = self.vocab_dict.get(word, 0)
            new_sentence.append(word_integer)
        new_sentence = self.pad_sentence(new_sentence, pad)
        return new_sentence

    def generate_word_num_vocabs(self):
        """
        This function returns two types of dictionaries
        first is where key is word and value is number
        and second is where key is number and value is word
        """

        self.vocab_dict_integer = {}
        for key, value in self.vocab_dict.items():
            self.vocab_dict_integer[value] = key

    def create_triplets(self, dataframe_path, n_rows):
        x, y = self._prepare_train_data(dataframe_path, n_rows)
        Q1, Q2 = x[0], x[1]
        all_samples = len(Q1)
        triplets = []
        for _ in range(all_samples):
            anchor = Q1.pop(0)
            positive = Q2.pop(0)
            anchor = np.array(anchor)
            positive = np.array(positive)
            triplets.append((anchor, positive))
        return triplets


preprocess = PreProcessData()
triplets = preprocess.create_triplets("train_quora.csv", 10000)
max_len = triplets[0][0].shape[0]

anchors = tf.convert_to_tensor([tf.reshape(i[0], (1, max_len)) for i in triplets])
positives = tf.convert_to_tensor([tf.reshape(i[1], (1, max_len)) for i in triplets])
del triplets

anchors = anchors[:10000]
positives = positives[:10000]


# This class encodes sentences into vectors so we can compute triplet loss later
class MeanNormalizationLayer(keras.layers.Layer):
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, mask=None, training=None, initial_state=None):
        return inputs / tf.math.sqrt(tf.math.reduce_sum(inputs * inputs, axis=-1, keepdims=True))


class CalculateMean(keras.layers.Layer):
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, mask=None, training=None, initial_state=None):
        return tf.math.reduce_mean(inputs, axis=1, keepdims=True)


class EmbeddingLayer:
    def __init__(self):
        self.MeanNormalizeLayer = MeanNormalizationLayer
        #self.CalculateMean = CalculateMean

    def build_model(self, vocab_size, d_model, units, input_len, batch_size):
        model = keras.Sequential(
            [
                layers.Embedding(input_dim=vocab_size,
                                 output_dim=d_model,
                                 input_length=input_len,
                                 input_shape=(1, input_len),
                                 batch_size=batch_size),
                layers.Reshape((input_len, d_model), input_shape=(1, batch_size, 1, input_len, d_model)),
                layers.LSTM(units=units, input_dim=(batch_size, input_len, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model), return_sequences=True),
                layers.LSTM(units=units, input_dim=(batch_size, d_model)),
                #self.CalculateMean(),  #  (32, 20)
                self.MeanNormalizeLayer()
            ]
        )
        return model


embed_layer = EmbeddingLayer()
vocab_size = 10000
d_model = 512
units = 100
batch_size = 32
model = embed_layer.build_model(vocab_size=vocab_size, d_model=d_model, units=units, input_len=anchors[0].shape[1],
                                batch_size=batch_size)

anchor_input = layers.Input(name="anchors", shape=(1, max_len), batch_size=32)
positive_input = layers.Input(name="positive", shape=(1, max_len), batch_size=32)

embedding_anchor = model(anchor_input)
embedding_positive = model(positive_input)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive], axis=1)

net = tf.keras.models.Model([anchor_input, positive_input], output)
net.summary()


@tf.autograph.experimental.do_not_convert
def triplet_loss(y_true, y_pred):
    """
    Triplet loss with negative hard mining
    """
    margin = 0.2
    anchor, positive = y_pred[:, :units], y_pred[:, units: units * 2]
    # new implementation
    scores = tf.linalg.matmul(anchor, tf.transpose(positive))
    positives = tf.linalg.tensor_diag_part(scores)
    negative_without_positive = scores - 2.0 * tf.eye(anchor.shape[0])
    closest_negative = tf.math.reduce_max(negative_without_positive, axis=1)
    negative_zero_on_duplicate = scores * (1.0 - tf.eye(batch_size))
    mean_negative = tf.math.reduce_sum(negative_zero_on_duplicate, axis=1) / (anchor.shape[0] - 1)
    triplet_loss_1 = tf.math.maximum(0.0, margin - positives + closest_negative)
    triplet_loss_2 = tf.math.maximum(0.0, margin - positives + mean_negative)
    triplet_loss = tf.math.reduce_mean(triplet_loss_1 + triplet_loss_2)
    return triplet_loss

class DataGenerator:
    def __init__(self, list_IDs, labels, batch_size=32, embedding_size=100, shuffle=True):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        [anchors[batch_size:batch_size],
             positives[i * batch_size:batch_size * (i + 1)]]

        y = np.empty((self.batch_size), dtype=int)

        x = [anchors[i * batch_size:batch_size * (i + 1)],
             positives[i * batch_size:batch_size * (i + 1)]]
        y = np.zeros((batch_size, 2 * embedding_size))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def data_generator(batch_size=32, embedding_size=100):
    i = 0
    batch_size

    while True:
        x = [anchors[i * batch_size:batch_size * (i + 1)],
             positives[i * batch_size:batch_size * (i + 1)]]
        y = np.zeros((batch_size, 2 * embedding_size))
        i += 1
        yield x, y

1000 // 32

batch_size = 32
epochs = 3
steps_per_epoch = int(anchors.shape[0] / batch_size)
net.compile(loss=triplet_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
net.fit(data_generator(batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs)
