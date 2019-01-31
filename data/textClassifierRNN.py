# author - Richard Liao
# Dec 26 2016
#
# updated by Raymond on Oct 22 2018 
# 
#  TODO:
#   add k-fold cross-validation by Raymond on Oct 30 2018
#  - Use Automatic Verification Datasets 
#  - Multi-GPU and distributed training
#    - (https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#exporting-a-model-with-tensorflow-serving)
#
import numpy as np
import pandas as pd
#import cPickle
#import cPickle as pickle
from collections import defaultdict
import re
import ast
import math
import string

import keras_metrics

from bs4 import BeautifulSoup

import sys
import os

#os.environ['KERAS_BACKEND']='theano'
os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.layers import Dropout


from keras import backend as K
from keras.engine.topology import Layer, InputSpec
#from keras import initializations
from keras import initializers

import matplotlib.pyplot as plt

#from get_data import get_labels
#from . import read_data_update
import read_data_update

#MAX_SEQUENCE_LENGTH = 1000
MAX_SEQUENCE_LENGTH = 100 #54 * 2 #300
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
#VALIDATION_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

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
data_train = pd.read_csv('~/Testground/data/imdb/labeledTrainData.tsv', sep='\t')
print(data_train.shape)

texts = []
labels = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data_train.sentiment[idx])
"""

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


def read_data(file):
    list_reviews = []
    asins = {}

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data = ast.literal_eval(str(line))
            """
            if 'reviews' in line:
                #line = line.replace("\'", "\"")
                reviews = data['reviews']      
               # len_review = 0
                #cnt = 0
                #mean_review = 0
                #max_len_review = 0
                for review in reviews:
                    list_reviews.append(review)
            """

            if 'tech' in line and len(data['tech']) > 0 and 'reviews' in line:
                _asin = str(data['asin'])

                reviews = data['reviews']
                
                for review in reviews:
                    list_reviews.append(review)
                

                params = data['tech']
                if len(params) > 0:
                    #asins[_asin] = [screensize,cpu, ram]
                    asins[_asin] = []
                    
                    for key, value in params.items():
                        if 'processor' in key.lower():
                            asins[_asin].append(value)
                        if 'ram' in key.lower():
                            asins[_asin].append(value)    
                        if 'screen size' in key.lower():
                            asins[_asin].append(value)
                        #hard drive
                        if 'hard' in key.lower():
                            asins[_asin].append(value)
                        # Graphics Coprocessor
                        if 'Graphics Coprocessor' in key.lower():
                            asins[_asin].append(value)

                        if len(asins[_asin]) == 5:
                            break

                    #asins[_asin].append(list_reviews)
                    asins[_asin].append(reviews)

    return asins

def train():

        # get documents
    """
    f1 = open('C:/Users/raymondzhao/myproject/dev.dplearning/data/amazon_data_0719.p', 'r')
    asins = pickle.load(f1)
    f1.close()
    """
    #dir = 'C:/Users/raymondzhao/myproject/dev.dplearning/data/'
    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    #dir = "/data/raymond/workspace/exp2/"
    file = dir + 'amazon_reviews.json'
    asins = read_data(file)

    file = dir + 'amazon_tech_all_5.csv'
    #df = pd.read_csv(file)

    # The text samples and their labels
    texts = []  #list of text samples
    #labels = array([])
    labels = [] #list of label ids
    labels_index = {}  # dictionary mapping label name to numeric id

    # ['14 inches', '2.16 GHz Intel Celeron', '4 GB SDRAM DDR3', 
    # [[b'I', b'placed', b'my', b'order', b'on', b'December', b'19th', b'and', b'was', b'under', b'the', b'impression', b'it', b'would', b'arrive', b'on', b'the', b'22nd']]
        # [screensize,cpu, ram, Hard Drive,Graphics Coprocessor, reviews]
    for _asin in asins:
        print("The asin %s:", _asin)
        # [screensize,cpu, ram, reviews]
        _cpu = asins[_asin][1]
        if _cpu:
           _cpu_id = get_cpu_label(_cpu)
           labels_index[_cpu] = _cpu_id

        _sscreen = asins[_asin][0]
        if _sscreen:
            _sscreen_id = get_sscreen_label(_sscreen)
            labels_index[_sscreen] = _sscreen_id

        _ram = asins[_asin][2]
        if _ram:
            _ram_id = get_ram_label(_ram)
            labels_index[_ram] = _ram_id
        
        _harddrive = asins[_asin][3]
        if _harddrive:
            _harddrive_id = get_harddrive_label(_harddrive)
            labels_index[_harddrive] = _harddrive_id
    
        #Graphics Coprocessor
        _graphprocessor = asins[_asin][4]
        if _graphprocessor:
            _graphprocessor_id = get_graphprocessor_label(_graphprocessor)
            labels_index[_graphprocessor] = _graphprocessor_id
        
        #reviews
        reviews = asins[_asin][5] 
        table = str.maketrans('', '', string.punctuation)
        #porter = PorterStemmer()
        for _t in reviews:
            # t =  " ".join(x.decode("utf-8") for x in _t) #bytes to str
            #words = text.split()
            # remove punctuation from each word , and stemming

            stripped = [w.decode("utf-8").lower().translate(table) for w in _t]  
            s = " ".join(x for x in stripped)
            #stripped = [w.decode("utf-8").translate(table) for w in _t] 
            #stripped = [w.decode("utf-8").lower().translate(table) for w in _t]
            #

            #t = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+','', t)
            texts.append(s)
            #labels.append(_cpu_id)
            #labels.append(_sscreen_id)
            #labels.append(_ram_id)
            labels.append(_harddrive_id)
            #labels.append(_graphprocessor_id)
    
    return texts, labels

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


def new_recall_score(y_pred,y_real):
    
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



def train_RNN():
    # get texts and labels
    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'

    #dir = '/data/raymond/workspace/exp2/'
    #file = dir +  'amazon_reviews.json'
    file = dir + 'amazon_reviews_copy.json'
    reviews = []
    asins_dict = read_data_update.get_amazon_texts_labels(file)
    # the generated asins
    generated_asins = read_data_update.read_generated_amazon_reviews()
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
texts, labels_matrix = train_RNN()
_labels = labels_matrix[:, 0].tolist()
_labels_PERCENT = labels_matrix[:, 0 : 14]
    
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
GLOVE_DIR = "C:/Users/raymondzhao/myproject/dev.deeplearning/data/"
embeddings_index = {}
f = open(GLOVE_DIR + 'glove.6B.100d.txt',  "r", encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

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

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
_preds = Dense(labels.shape[1], activation='softmax')(l_lstm)
preds = Dropout(2)(_preds)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])

# summarize the model
print("model fitting - Bidirectional LSTM")
print(model.summary())

# fit the model
#epochs = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
#epochs = [3, 4,5, 6, 7, 10]
#ssreen[0.5070845986019322, 0.48973507110125514, 0.48719953534089894, 0.5122896375063295, 0.49724417154941686, 0.4466746164046027, 0.4621094331754684, 0.47069590234449604, 0.4701784171804724, 0.45380365174396087, 0.41863848926220476]

fs = []
acc = []
pre = []
rec = []
bsize = 64
#epoch = int(len(x_train) / bsize)
#epoch = 12
for epoch in range(5):
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
        recall = new_recall_score (y_pred, _labels_PERCENT)
        recalls.append(recall)
        print(recall)

    """
    print(scores.keys())
    print(scores['test_precision_macro'])
    print(scores['test_recall_macro'])

    """
    print(np.mean([precisions, recalls], 0))

"""
print("The metric \n")
print('accuracy: ',acc)
print('precision: ',pre)
print('recall: ',rec)
print('f1: ',fs)
"""

"""
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
"""

print("Done")

"""
# Attention GRU network		  
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        #eij = K.tanh(K.dot(x, self.W))
        eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
        
        ai = K.exp(eij)
        #weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        #weighted_input = x*weights.dimshuffle(0,1,'x')
        #return weighted_input.sum(axis=1)
        
        weights = ai/K.expand_dims(K.sum(ai, axis=1),1)

        weighted_input = x*K.expand_dims(weights,2)
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
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

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer()(l_gru)
preds = Dense(labels.shape[1], activation='softmax')(l_att)
model = Model(sequence_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', keras_metrics.precision(), keras_metrics.recall()])

# summarize the model
print(model.summary())

# fit the model
hist = model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_split=0.1)

print(hist.history)

# evaluate the model
#print('recall + precision: %s' % (_metrics.get_data() * 100))
    
loss, accuracy, precision, recall = model.evaluate(x_val, y_val, verbose=0)
print('metrics: %f' % accuracy)


"""
