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
#import keras_metrics

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

import math

#
#import read_data
import read_data_update_copy


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


def get_cpu_label(_str):
    # [ 2 GHz AMD A Series, 1.1 GHz Intel Celeron, 2.16 GHz Intel Celeron,3 GHz 8032,1.6 GHz,3.5 GHz 8032,4 GHz Intel Core i7]
    _cpu_map = {
        "amd": 0,
        "1.1 Intel":1,
        "1.5-2.5 Intel": 2,
        "2.5-3.5 Intel":3,
        "3.5 Intel":4,
        "others":5
         
    }

    #_cpu_label = 4 #unknown
    if 'amd' in _str.lower():
        _cpu_label = 0
    else: #Intel
        _cpu_frequency = float(re.search('[\d]+[.\d]*', _str).group())   
        if _cpu_frequency <= 1.5:
            _cpu_label = 1
        elif _cpu_frequency <= 2.5:
            _cpu_label = 2
        elif _cpu_frequency <= 3:
            _cpu_label = 3
        elif _cpu_frequency <= 3.5:
            _cpu_label = 4
        elif _cpu_frequency > 3.5:
            _cpu_label = 5
        
        #_cpu_label = 1

    return _cpu_label

def get_sscreen_label(_str):
    # [ 11.6 inches, 13.3 inches,14 inches,15.6 inches, 17.3 inches ]
    _sscreen_map = {
        "<= 12 inches": 0,
        "<= 13 inches":1,
        "<= 14 inches": 2,
        "<= 15 inches":3,
        "> 15 inches":4
    }

    _sscreen_label = 4 #unknown
    if 'inches' in _str.lower():
        _sscreen_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
        if _sscreen_size <= 12:
            _sscreen_label = 0
        elif _sscreen_size <= 13:
            _sscreen_label = 1
        elif _sscreen_size  <= 14:
            _sscreen_label = 2
        elif _sscreen_size <= 15:
            _sscreen_label = 3
        else:
            _sscreen_label = 4

    return _sscreen_label


def get_ram_label(_str):
    # [ "4 GB SDRAM DDR3", "4 GB DDR3 SDRAM","8 GB",4 GB SDRAM DDR4","16 GB DDR4" ,"2 GB SDRAM","6 GB DDR SDRAM", "12 GB DDR SDRAM" ]
    _ram_map = {
        "2 GB SDRAM": 0,
        "4 GB SDRAM DDR3": 1,
        "6 GB DDR SDRAM":2,
        "8 GB SDRAM DDR3": 3,
        "8 GB SDRAM DDR4": 3,
        "12 GB DDR SDRAM":4,
        "16 GB DDR4" :5,
        "others":6,
    }

    #_ram_label = 7 #unknown
    if 'gb'  in _str.lower():
        _ram_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
        if _ram_size == 2:
            _ram_label = 0
        elif _ram_size == 4:
            _ram_label = 1
        elif _ram_size  == 6:
            _ram_label = 2
        elif _ram_size == 8:
            if 'ddr3' in _str.lower():
                _ram_label = 3
            elif 'ddr4' in _str.lower():
                _ram_label = 3
            else:
                _ram_label = 3
        elif _ram_size  == 12:
            _ram_label = 4
        elif _ram_size  == 16:
            _ram_label = 5
        else:
            #_ram_label = 7
            pass

    return _ram_label

def get_harddrive_label(_str):
    # [ '16 GB SSD', '128 GB SSD', '1 TB HDD 5400 rpm', 
    # '256 GB Flash Memory Solid State', '500 GB HDD 5400 rpm', 
    # 'Flash Memory Solid State', '1000 GB Hybrid Drive',
    # '2 TB HDD 5400 rpm', '32 GB SSD','64 GB SSD'
    #
    # ]
    _harddrive_map = {
        "SSD <= 128": 0,
        "SSD > 128": 1,
        "HDD > 1T" :2,
        "HDD ~= 1T" :3,
        "HDD ~= 500G" :4,
        "HDD < 500G" :5,
        "others": 5
    }

    #_harddrive_label = 5 #unknown
    if 'ssd' or 'solid' or 'mechanical' in _str.lower():
        if num_there(_str):
            _harddrive_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
            if _harddrive_size <= 128:
                _harddrive_label = 0
            else:
                _harddrive_label = 0
        else:
            _harddrive_label = 0

    if 'hdd' in _str.lower():
        #_harddrive_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
        if 'tb' in _str.lower():
            if num_there(_str):
                _harddrive_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
                if _harddrive_size > 1:
                    _harddrive_label = 1
                else:
                    _harddrive_label = 2
        else:
            if num_there(_str):
                _harddrive_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
                if _harddrive_size >= 500:
                    _harddrive_label = 3
                else:
                    _harddrive_label = 4

    return _harddrive_label


def num_there(s):
    return any(i.isdigit() for i in s)

def get_graphprocessor_label(_str):
    """
    [ 'Intel HD Graphics 500', 'Intel HD Graphics 505', 'Intel UHD Graphics 620',
    'AMD', 'NVIDIA GeForce GTX 1050', 'GTX 1050 Ti'
      'PC', 'FirePro W4190M - AMD', 'Integrated', 

    ]
    """
    _graphprocessor_map = {
        "Intel HD Graphics 50X": 0,
        "Intel HD Graphics 505": 0,
        "Intel UHD Graphics 620":1,
        "Intel HD Graphics" :1,
        "AMD Radeon R2": 3,
        "AMD Radeon R5": 4,
        "AMD Radeon R7": 4,
        "AMD Radeon R4" :3,
        "NVIDIA GeForce GTX 1050": 58,
        "NVIDIA GeForce 940MX" :  5,
        "Integrated" : 6,
        "others| PC | FirePro W4190M ": 7
    }

    _graphprocessor_label = 5 #unknown
    if 'intel' in _str.lower():
        
        if num_there(_str):
            _graphprocessor_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
            if _graphprocessor_size == 500:
                _graphprocessor_label = 0
            elif _graphprocessor_size == 505:
                _graphprocessor_label = 0
            elif _graphprocessor_size  == 620:
                _graphprocessor_label = 0
            else:
                _graphprocessor_label = 0

    if 'amd' in _str.lower():        
        if 'r2' in _str.lower():
            _graphprocessor_label = 1
        if 'r5' in _str.lower():
            _graphprocessor_label = 2
        if 'r7' in _str.lower():
            _graphprocessor_label =2
        if 'r4' in _str.lower():
            _graphprocessor_label = 1
        

    if 'nvidia' in _str.lower():
        if num_there(_str):
            _graphprocessor_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
            if _graphprocessor_size == 1050:
                _graphprocessor_label = 3
            if _graphprocessor_size == 940:
                _graphprocessor_label = 3  

    if 'integrated' in _str.lower():
        _graphprocessor_label  = 4


    return _graphprocessor_label


def _precision_score(y_test, y_pred, K):
    precision = 0

    num = 0
    i = 0

    """
    while i < len(y_test):
        lst = y_test[i].tolist()
        if y_pred[i] in lst:
            num = num + 1

        i = i + 1

    precision = num / (len(y_pred) * y_test.shape[1])
    """

    while i < y_test.shape[1]:
        lst = y_test[:, i].tolist()
        j = 0
        while  j < len(lst):
            if lst[j] in y_pred[j].tolist():
                num = num + 1
            j = j + 1

        i = i + 1
    
    precision = num / (len(y_pred) * y_test.shape[1])
   
    
    """
    i = set(y_test.flatten()).intersection(y_pred.flatten())
    len1 = len(y_pred)
    if len1 == 0:
        precision = 0
    else:
        precision =  len(i) / len1
    """
    return precision


def new_recall_score(y_pred,y_real, K):
    
    recall = 0
    total_recall=0
    num = y_pred.shape[0]
    
#    for each row, if there is matched class, then it's a count
#    simply check each row of y_pred and y_real if their intersection is true
    for i in range(y_pred.shape[0]):
        if(set(y_pred[i,:]).intersection(y_real[i,:])):
            total_recall=total_recall+1
            
    recall=total_recall/num
    return recall


def _recall_score(y_test, y_pred, K):
    recall = 0

    num = 0
    i = 0
    """
    while i < len(y_test):
        lst = y_test[i].tolist()
        if y_pred[i] in lst:
            num = num + 1

        i = i + 1
    """

    
    while i < y_test.shape[1]:
        lst = y_test[:, i].tolist()
        j = 0
        while  j < len(lst):
            if lst[j] in y_pred[j].tolist():
                num = num + 1
            j = j + 1

        i = i + 1
    
    recall = num / (len(y_test) * K)
    
    """
    i = set(y_test).intersection(y_pred)
    return len(i) / len(y_test)
    """

    return recall
   
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

def train_HAN():
    # get texts and labels
    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'

    #dir = '/data/raymond/workspace/exp6/'
    #file = dir +  'amazon_reviews.json'
    file = dir + 'amazon_reviews_copy.json'
    reviews = []
    asins_dict = read_data_update_copy.get_amazon_texts_labels(file)
    # the generated asins
    generated_asins = read_data_update_copy.read_generated_amazon_reviews()
    texts = []
    #generated_texts = []
    labels_lst = []
    for asin in asins_dict:
        """
        for text in asins_dict[asin][0]:
            texts.append(text)
        """
        #i = 0
        if asin in generated_asins:
            i = 0
            while i < len(generated_asins[asin]):
                texts.append(generated_asins[asin][i])
                labels_lst.append(asins_dict[asin][1][i])

                i = i+1

            """
            for text in generated_asins[asin]:
                texts.append(text)

            for label in asins_dict[asin][1]:
                labels_lst.append(label)
            """
        else:
            for text in asins_dict[asin][0]:
                texts.append(text)
        
        #texts.append(asins_dict[asin][0])
            for label in asins_dict[asin][1]:
                labels_lst.append(label)

    #
    labels_matrix = np.array(labels_lst)

    PERCENT = 0.1 # the range between the pred object
    #_labels = labels_matrix[:, 0].tolist()
    #_labels_PERCENT = labels_matrix[:, 0 : 7 * 1]

    return texts, labels_matrix

#
#texts, labels = train()
texts, labels_matrix = train_HAN()
_labels = labels_matrix[:, 0].tolist()
_labels_PERCENT = labels_matrix[:, 0 : math.ceil(labels_matrix.shape[1] * 0.2)]
    
#
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int((VALIDATION_SPLIT + TEST_SPLIT) * data.shape[0])
#nb_validation_samples = int(VALIDATION_SPLIT *  data.shape[0])
nb_test_samples = int(TEST_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_test_samples:]
y_test = labels[-nb_test_samples:]
_y_test = _labels[-nb_test_samples:]
_y_test_PERCENT =_labels_PERCENT[-nb_test_samples:]
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
#GLOVE_DIR = "/data/raymond/workspace/exp2/"
#GLOVE_DIR = "C:/Users/raymondzhao/myproject/dev.deeplearning/data/"
#GLOVE_DIR = "/data/raymond/workspace/exp6/"
GLOVE_DIR = "C:/Users/raymondzhao/myproject/dev.deeplearning/nsrc/"
embeddings_index = {}
f = open(GLOVE_DIR + 'glove.6B.100d.txt',  "r", encoding="utf-8")
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
              optimizer='adam',
              metrics=['accuracy'])

print("model fitting - Hierachical attention network")

fs = []
acc = []
pre = []
rec = []
bsize = 64
#epoch = int(len(x_train) / bsize)
#epoch = 12
for epoch in range(25):
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

    #loss, accuracy, precision, recall = model.evaluate(x_test, y_test, batch_size=bsize, verbose=0)
    
    """
    f1 = (precision + recall) / 2
    fs.append(f1)
    acc.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    print('accuracy: %f, precision: %f, recall: %f, f1: %f' % (accuracy, precision, recall, f1))
    """

    #predict
    y_proba = model.predict(x_test)
    y_proba_ind = np.argsort(-y_proba)
    print("y_proba: ", y_proba_ind)

    K = 5

    #num = 0
    #y_test = y_lst[:, 0]
    y_pred_max = y_proba_ind[:, 0]

    #precision_1 = metrics.precision_score(_y_test, y_pred_max, average='macro')
    #print("precision_1: %f" % precision_1)
        
    #num_lst = []
 
    #num = np.sum(y_test == y_pred_max)
    #num_lst.append(num)
    
    print("precision:")
    i = 0
    precisions = []
    
    while i < K:
        #y_test = y_lst[:, 0:i+1]
        y_pred = y_proba_ind[:, 0:i+1]

        i = i+1

        precision = _precision_score(y_pred, _y_test_PERCENT, K)
        precisions.append(precision)

        print(precision)

    print("recall:")
    #recall = []
    #recall_1 = metrics.recall_score(_y_test, y_pred_max, average='micro')
    #recall.append(recall_1)
    #print("recall_1: %f" % recall_1)
    
    i = 0
    recalls = []
    while i < K:
        #y_test = y_lst[:, 0:i+1]
        y_pred = y_proba_ind[:, 0:i+1]

        i = i+1

        #recall = _recall_score(y_pred, _y_test_PERCENT, K)
        recall = new_recall_score(y_pred, _labels_PERCENT, K)
        recalls.append(recall)
        print(recall)

"""
print("The metric \n")
print('accuracy: ',acc)
print('precision: ',pre)
print('recall: ',rec)
print('f1: ',fs)
"""