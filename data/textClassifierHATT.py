# author - Richard Liao
# Dec 26 2016
#
# updated by raymond
# - data-preprocessing 
# 
# 
# 
import numpy as np
import pandas as pd
#import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
os.environ['KERAS_BACKEND']='tensorflow'

import string
import keras_metrics

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints

import keras_metrics

from nltk import tokenize

#
import read_data

#MAX_SEQUENCE_LENGTH = 1000
MAX_SEQUENCE_LENGTH = 100 #300
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
#VALIDATION_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

#

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

"""
data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')
print(data_train.shape)

from nltk import tokenize

reviews = []
labels = []
texts = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    text = clean_str(text.get_text().encode('ascii', 'ignore'))
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)

    labels.append(data_train.sentiment[idx])
"""

# get texts and labels
dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
#dir = '/data/raymond/workspace/exp2/'
file = 'amazon_reviews.json'
reviews = []
texts, label_inds = read_data.get_amazon_texts_labels(file)

#
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
#nb_validation_samples = int((VALIDATION_SPLIT + TEST_SPLIT) * data.shape[0])
nb_validation_samples = int(VALIDATION_SPLIT *  data.shape[0])
nb_test_samples = int(TEST_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
"""
x_val = data[-nb_validation_samples:-nb_test_samples]
y_val = labels[-nb_validation_samples:-nb_test_samples]

x_test = data[-nb_test_samples:]
y_test = labels[-nb_test_samples:]
"""

print('Traing and validation set number of positive and negative reviews')
print(y_train.sum(axis=0))
print(y_test.sum(axis=0))

#GLOVE_DIR = "~/Testground/data/glove"
GLOVE_DIR = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
#GLOVE_DIR = "/data/raymond/workspace/exp2/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
#embedding_matrix = np.random.uniform(-0.001, 0.001, (len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


class Attention(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(Attention, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        #uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

"""
class Attention(Layer):
    def __init__(self, step_dim,attention_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.attention_dim = attention_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        self.features_dim = input_shape[-1]
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
 
        #super(AttLayer, self).build(input_shape)
        self.trainable_weights = [self.W, self.b, self.u]

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
"""
#
sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
l_att = Attention(MAX_SEQUENCE_LENGTH)(l_lstm)
#l_att = Flatten()(l_att)
preds = Dense(labels.shape[1], activation='softmax')(l_att)

model = Model(sentence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])

print("model fitting - Hierachical attention network")

fs = []
acc = []
pre = []
rec = []
bsize = 64
#epoch = int(len(x_train) / bsize)
#epoch = 12
for epoch in range(10):
    hist = model.fit(x_train, y_train, 
                     validation_split=VALIDATION_SPLIT,
                     #validation_data=(x_val, y_val),
                     epochs=epoch,
                     batch_size=bsize)
    print(hist.history)

    # evaluate the model 
    """
    if len(hist.history) > 0:
        val_acc.append(hist.history['val_acc'][-1])
        val_pre.append(hist.history['val_precision'][-1])
        val_rec.append(hist.history['val_recall'][-1])
        #val_f1 = (hist.history['val_precision'][-1] + hist.history['val_recall'][-1]) / 2
        val_f1.append((hist.history['val_precision'][-1] + hist.history['val_recall'][-1]) / 2)
    """

    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, batch_size=bsize, verbose=0)
    
    f1 = (precision + recall) / 2
    fs.append(f1)
    acc.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    print('accuracy: %f, precision: %f, recall: %f, f1: %f' % (accuracy, precision, recall, f1))

print("The metric \n")
print('accuracy: ',acc)
print('precision: ',pre)
print('recall: ',rec)
print('f1: ',fs)