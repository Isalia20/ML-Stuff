import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
import tensorflow as tf


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
        stop_words_eng = stopwords.words('english')
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
                if word_processed_1 != "" and word_processed_1 not in stop_words_eng:
                    Q1_sentence.append(word_processed_1)

                word_processed_2 = ps.stem(word_2.lower())
                word_processed_2 = word_processed_2.translate(str.maketrans('', '', string.punctuation))
                if word_processed_2 != "" and word_processed_2 not in stop_words_eng:
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
                    # Skipping the null word
                    if word == "":
                        continue

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
            anchor = anchor
            positive = positive
            triplets.append((anchor, positive))
        return triplets


def data_generator(anchors_input, positives_input, batch_size_input=32, embedding_size=100):
    x_size = anchors_input.shape[0]

    while True:
        x_ids = np.random.choice(x_size, batch_size_input)
        x = [tf.gather(anchors_input, indices=x_ids), tf.gather(positives_input, indices=x_ids)]
        y = np.zeros((batch_size_input, 2 * embedding_size))
        yield x, y
