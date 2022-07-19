#WIP
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
        self.vocab_dict_1 = None
        self.vocab_dict_2 = None
        # Vocabulary dictionary where key is integer and values are words
        self.vocab_dict_integer_1 = None
        self.vocab_dict_integer_2 = None
        self.max_len = None

    def _process_dataframe(self, dataframe_path):
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

    @staticmethod
    def get_max_sentence_len(Q1_processed, Q2_processed):
        Q_processed = Q1_processed + Q2_processed
        max_len = 0
        for sentence in Q_processed:
            sentence_len = len(sentence)
            if sentence_len > max_len:
                max_len = sentence_len
        return max_len

    @staticmethod
    def pad_sentence(sentence, pad, max_len):
        sentence_len = len(sentence)
        max_len = 2 ** (int(np.log2(max_len)) + 1)  # For better computations on GPU
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

        vocab_dict_q1 = {}
        vocab_dict_q2 = {}

        vocab_sets = [vocab_q1, vocab_q2]
        vocab_dicts = [vocab_dict_q1, vocab_dict_q2]

        for vocab_set, vocab_dict in zip(vocab_sets, vocab_dicts):
            # We start from 1 due to reason of adding padding tokens equal 0
            for number, items in enumerate(vocab_set, 1):
                word = items
                vocab_dict[word] = number

        return vocab_dict_q1, vocab_dict_q2

    def _prepare_train_data(self, dataframe_path, n_rows):
        Q1_processed, Q2_processed, is_duplicate = self._preprocess_data(dataframe_path, n_rows)
        max_len = self.get_max_sentence_len(Q1_processed, Q2_processed)
        self.vocab_dict_1, self.vocab_dict_2 = self._create_word_indices(Q1_processed, Q2_processed)
        y = is_duplicate  # x is defined later
        Q1_integers = []
        Q2_integers = []
        questions = [Q1_processed, Q2_processed]
        questions_integers = [Q1_integers, Q2_integers]
        vocab_dicts = [self.vocab_dict_1, self.vocab_dict_2]
        for question_index in range(len(questions)):
            vocab_dict_temp = vocab_dicts[question_index]
            for sentence in questions[question_index]:
                word_nums = []
                for word in sentence:
                    word_num = vocab_dict_temp.get(word)
                    word_nums.append(word_num)
                # (sentence, pad, max_len)
                word_nums = self.pad_sentence(word_nums, 0, max_len)
                question_integers_list = questions_integers[question_index]
                question_integers_list.append(word_nums)
        x = questions_integers
        return x, y

    def _generate_word_num_vocabs(self):
        """
        This function returns two types of dictionaries
        first is where key is word and value is number
        and second is where key is number and value is word
        """
        vocab_dicts = [self.vocab_dict_1, self.vocab_dict_2]
        num_dicts = [self.vocab_dict_integer_1, self.vocab_dict_integer_2]

        for vocab_dict_index in range(len(vocab_dicts)):
            vocab_dict = vocab_dicts[vocab_dict_index]
            num_dict = num_dicts[vocab_dict_index]
            for key, value in vocab_dict.items():
                num_dict[value] = key

    def create_triplets(self, dataframe_path, n_rows):
        x, y = self._prepare_train_data(dataframe_path, n_rows)
        Q1, Q2 = x[0], x[1]
        all_samples = len(Q1)
        triplets = []
        for _ in range(all_samples):
            anchor = Q1.pop(0)
            positive = Q2.pop(0)
            rand_negative = np.random.choice(2)
            if rand_negative == 0:
                try:
                    random_sample = np.random.choice(len(Q1))
                    negative = Q1[random_sample]
                except ValueError:
                    continue
            else:
                try:
                    random_sample = np.random.choice(len(Q2))
                    negative = Q2[random_sample]
                except ValueError:
                    continue
            anchor = np.array(anchor)
            positive = np.array(positive)
            negative = np.array(negative)
            triplets.append((anchor, positive, negative))
        return triplets


class DistanceLayer:

    def dist_call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance


preprocess = PreProcessData()
triplets = preprocess.create_triplets("train_quora.csv", 1000000)

anchors = tf.convert_to_tensor([tf.reshape(i[0], (1, 128)) for i in triplets])
positives = tf.convert_to_tensor([tf.reshape(i[1], (1, 128)) for i in triplets])
negatives = tf.convert_to_tensor([tf.reshape(i[2], (1, 128)) for i in triplets])
del triplets

anchors = anchors[:10000]
positives = positives[:10000]
negatives = negatives[:10000]

# This class encodes sentences into vectors so we can compute triplet loss later
class EmbeddingLayer:
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
                layers.LSTM(units=units, input_dim=(batch_size, d_model)),
            ]
        )
        return model



embed_layer = EmbeddingLayer()
vocab_size = 10000
d_model = 512
units = 20
batch_size = 32
model = embed_layer.build_model(vocab_size=vocab_size, d_model=d_model, units=units, input_len=anchors[0].shape[1],
                                batch_size=batch_size)

anchor_input = layers.Input(name="anchors", shape=(1, 128), batch_size=32)
positive_input = layers.Input(name="positive", shape=(1, 128), batch_size=32)
negative_input = layers.Input(name="negative", shape=(1, 128), batch_size=32)

embedding_anchor = model(anchor_input)
embedding_positive = model(positive_input)
embedding_negative = model(negative_input)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model([anchor_input, positive_input, negative_input], output)
net.summary()

@tf.autograph.experimental.do_not_convert
def triplet_loss(y_true, y_pred):
    alpha = 0.2

    anchor, positive, negative = y_pred[:, :units], y_pred[:, units: units * 2], y_pred[:, units * 2:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)

def data_generator(batch_size=32, embedding_size=100):
    i = 0
    batch_size

    while True:
        x = [anchors[i * batch_size:batch_size * (i + 1)],
             positives[i * batch_size : batch_size * (i + 1)],
             negatives[i * batch_size : batch_size * (i + 1)]]
        y = np.zeros((batch_size, 3 * embedding_size))
        yield x, y

batch_size = 32
epochs = 10
steps_per_epoch = int(anchors.shape[0]/batch_size)
net.compile(loss=triplet_loss, optimizer='adam')

_ = net.fit(
    data_generator(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs
)
