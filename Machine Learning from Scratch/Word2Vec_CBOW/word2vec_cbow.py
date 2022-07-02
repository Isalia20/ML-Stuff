import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import string
import numpy as np

question_pairs_train = pd.read_csv("train_quora.csv")

Q1 = question_pairs_train["question1"].to_numpy()
Q2 = question_pairs_train["question2"].to_numpy()

# I only select 5000 here as if I select more it causes memory errors
Corpus = (Q1[:10000], Q2[:10000])

ps = PorterStemmer()


class Word2Vec_CBOW():
    def __init__(self):
        self.ps = ps
        self.model = None
        self.data = None
        self.vocab = None

    # Pass data to this function as a tuple
    def process_data(self, corpus):
        Q1 = corpus[0]
        Q2 = corpus[1]
        Q1_processed = []
        Q2_processed = []
        for sentence_1, sentence_2 in zip(Q1, Q2):
            # initializing lists for appending processed sentences
            Q1_sentence = []
            Q2_sentence = []
            try:
                sentence_1_tokenized = nltk.word_tokenize(sentence_1)
                sentence_2_tokenized = nltk.word_tokenize(sentence_2)
            except TypeError:
                continue
            for word_1, word_2 in zip(sentence_1_tokenized, sentence_2_tokenized):
                word_processed_1 = self.ps.stem(word_1.lower())
                word_processed_1 = word_processed_1.translate(str.maketrans('', '', string.punctuation))
                Q1_sentence.append(word_processed_1)

                word_processed_2 = self.ps.stem(word_2.lower())
                word_processed_2 = word_processed_2.translate(str.maketrans('', '', string.punctuation))
                Q2_sentence.append(word_processed_2)

            Q1_processed.append(Q1_sentence)
            Q2_processed.append(Q2_sentence)

        processed_corpus = Q1_processed + Q2_processed

        return processed_corpus

    # This method is inefficient but just trying to build this quickly and then iterate
    def create_vocab(self, corpus):
        processed_corpus = self.process_data(corpus=corpus)
        vocab = {}

        for sentence in processed_corpus:
            for word in sentence:
                if vocab.get(word, 0) == 0:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        vocab_final = {}
        for number, items in enumerate(vocab.items()):
            word = items[0]
            vocab_final[word] = number

        return vocab_final

    @staticmethod
    # Generator Function for generating context and target words
    def get_windows(sentence, C):
        # Here C stands for the amount of words that should be in the context(before or after)
        # So if it is let's say 2 then in the sentence "I like Mozzarella and other types of cheese"
        # for the word Mozzarella context words will be: "I like" "and other"
        i = C
        while i < len(sentence) + C:
            #TODO start tokens and end tokens to remove try except here
            try:
                center_word = sentence[i]
                context_words = sentence[(i - C): i] + sentence[(i + 1):(i + C + 1)]
                yield context_words, center_word
                i += 1
            except IndexError:
                i += 1
                continue

    def generate_train_data(self, corpus, C):
        processed_corpus = self.process_data(corpus)
        target_context_dict = {}
        for sentence in processed_corpus:
            for context_target in self.get_windows(sentence, C):
                context, target = context_target[0], context_target[1]
                context = tuple(context)
                if target_context_dict.get((target, context), 0) == 0:
                    target_context_dict[(target, context)] = 1
                else:
                    target_context_dict[(target, context)] += 1
        return target_context_dict

    def generate_one_hot_vector(self, word):
        one_hot_vector = np.zeros([1, len(self.vocab)])
        one_hot_vector[:, self.vocab[word]] = 1
        return one_hot_vector

    def prepare_train_data(self, corpus, C):
        self.vocab = self.create_vocab(corpus)
        target_context_dict = self.generate_train_data(corpus, C)
        x = []
        y = []
        print("started preparing train data")
        for target, context in target_context_dict.keys():
            one_hot_vectors = []
            for word in context:
                one_hot_vector = self.generate_one_hot_vector(word)
                one_hot_vectors.append(one_hot_vector)

            one_hot_vectors = np.array(one_hot_vectors).reshape(-1, len(self.vocab))
            # Cast them into a matrix form
            try:
                one_hot_vectors = np.mat(one_hot_vectors)
                context_representation = np.mean(one_hot_vectors, axis=0)
                x.append(context_representation)
                # X is done here now we append to y the target word one hot vector
                y_one_hot = self.generate_one_hot_vector(target)
                y.append(y_one_hot)
            except ValueError:
                continue

        # Transpose x and y to get the correct table for representation

        # return x,y, self.vocab
        x = np.mat(np.array(x).reshape(-1, len(self.vocab)))
        y = np.mat(np.array(y).reshape(-1, len(word2vec.vocab)))
        return x, y, self.vocab

    # n_dimensions is here a parameter we want our embedded vectors to be represented with
    def build_model(self, n_dimensions):
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model = keras.Sequential()
        model.add(layers.Dense(n_dimensions, activation="relu", name="embedding_layer"))
        model.add(layers.Dense(len(self.vocab), activation="softmax", name="prediction_layer"))
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def train_model(self, x, y, epochs=1000, batch_size=32):
        model = self.build_model(n_dimensions=300)
        model.fit(x, y, epochs=epochs, batch_size=batch_size)
        self.model = model

word2vec = Word2Vec_CBOW()

X, y, vocab = word2vec.prepare_train_data(Corpus, 2)

# Remove some lists to free up memory
del(Q1)
del(Q2)
del(question_pairs_train)

word2vec.train_model(X, y, 100, 32)
