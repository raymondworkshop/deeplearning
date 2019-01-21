# based on GloVe word embeddings
# @raymond
#
import numpy
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.initializers import Constant
from keras.layers import Input, GlobalMaxPooling1D
from keras.utils import to_categorical


from keras.models import Model
import tensorflow as tf
from keras.layers import Lambda


from keras import regularizers

from keras import backend as K

#from engine_training import ExtendedModel
#from engine_training import ThetaPrime
import keras_metrics

import pickle
import pandas as pd

import re
import ast
import math

#import nltk
import string
from nltk.stem import PorterStemmer

from sklearn import svm 
from sklearn.metrics import recall_score
#from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics


#
#from . import read_data_update
import read_data_update

#
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 54 * 2 #300
EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

amazon_refurbished_goods_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                     21, 22, 23, 24, 26, 28, 31, 32, 33, 34, 38, 40, 42, 43, 44, 45,
                     48, 50, 51, 53, 56, 57, 58, 59, 60, 62, 65, 69,70, 72,
                     75, 76, 78, 79, 81, 82, 83, 84, 85, 87, 89, 90, 91, 93,
                     94, 95, 96, 97, 99, 102, 103, 105, 106]


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


def _read_data(file):
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

def aver_emb_encoder(x_emb, x_mask): # emb / L
    """ compute the average over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
    H_enc_0 = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc_0, [1, 3])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc

def max_emb_encoder(x_emb, x_mask):
    """ compute the max over every dimension of word embeddings """
    x_mask_1 = tf.expand_dims(x_mask, axis=-1) # L 1
    x_mask_1 = tf.expand_dims(x_mask_1, axis=-1)  # L 1 1
    H_enc = tf.nn.max_pool(tf.multiply(x_emb, x_mask_1), [1, MAX_SEQUENCE_LENGTH, 1, 1], [1, 1, 1, 1], 'VALID')
    H_enc = tf.squeeze(H_enc)

    return H_enc

def concat_emb_encoder(x_emb, x_mask, opt):
    """ concat both the average and max over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
    H_enc = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc, [1, 3])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1

    H_enc_1 = H_enc / x_mask_sum  # batch emb

    H_enc_2 = tf.nn.max_pool(x_emb, [1, opt.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
    H_enc_2 = tf.squeeze(H_enc_2, [1, 3])

    H_enc = tf.concat([H_enc_1, H_enc_2], 1)

    return H_enc


def _precision_score(y_test, y_pred, K):
    precision = 0

    num = 0
    i = 0
    
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


def _recall_score(y_test, y_pred):
    recall = 0

    num = 0
    i = 0
    while i < len(y_test):
        lst = y_test[i].tolist()
        if y_pred[i] in lst:
            num = num + 1

        i = i + 1
    
    recall = num / len(y_test)

    #i = set(y_test).intersection(y_pred)
    #return len(i) / len(y_test)

    return recall


def train_wordembedding():
    # get documents
    """
    f1 = open('C:/Users/raymondzhao/myproject/dev.dplearning/data/amazon_data_0719.p', 'r')
    asins = pickle.load(f1)
    f1.close()
    """

    """
    #dir = 'C:/Users/raymondzhao/myproject/dev.dplearning/data/'
    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    #dir = "/data/raymond/workspace/exp2/"
    #file = dir + 'amazon_reviews.json'
    file = dir + 'amazon_reviews_copy.json'
    asins = _read_data(file)

    file = dir + 'amazon_tech_all_5.csv'
    #df = pd.read_csv(file)
    """
    
    """
    asins = {}
    lst1 = ['14 inches', '2.16 GHz Intel Celeron', '4 GB SDRAM DDR3', [['I', 'placed', 'my', 'order', 'on', 'December', '19th'], ['I', 'just', 'received', 'this']] ]
    lst2 = ["15.6 inches", "3.3 GHz Intel Core i5", "8 GB DDR4 2666MHz", [['20th', 'on', 'Nov' ], [ 'this','just', 'went']] ] 
    lst3 = ["15.6 inches", "2.4 GHz AMD A Series", "8 GB SDRAM DDR4", [[b'very', b'hard', b'to', b'go', b'wrong,'], [b'Mine', b'just', b'came' ]] ] 

    #[ ['I', 'placed', 'my', 'order', 'on', 'December', '19th'], ['I', 'just', 'received', 'this', 'laptop'] ]

    asins["B07193JRJR"] = lst1
    asins["B07BP9QG2J"] = lst2
    asins[ "B07BLKT38D"] = lst3
    """

    """
    #'14 inches', '2.16 GHz Intel Celeron', '4 GB SDRAM DDR3', '16 GB SSD', 'Intel HD Graphics', 'Intel HD Graphics',

    # The text samples and their labels
    texts = []  #list of text samples
    #labels = array([])
    _labels = [] #list of label ids
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
            _labels.append(_cpu_id)
            #_labels.append(_sscreen_id)
            #_labels.append(_ram_id)
            #_labels.append(_harddrive_id)
            #_labels.append(_graphprocessor_id)

    """

    """
    _cpus = df.loc[1, :].tolist()
    for _cpu in _cpus[1:]:
        #_cpu = asins[_asin][1]
        if _cpu:
           _cpu_id = get_cpu_label(_cpu)
           labels_index[_cpu] = _cpu_id

    _sscreens = df.loc[0, :].tolist()
    for _sscreen in _sscreens[1:]:
        #_sscreen = asins[_asin][0]
        if _sscreen:
            _sscreen_id = get_sscreen_label(_sscreen)
            labels_index[_sscreen] = _sscreen_id
    
    _rams = df.loc[2, :].tolist()
    for _ram in _rams[1:]:
        if _ram:
            _ram_id = get_ram_label(_ram)
            labels_index[_ram] = _ram_id
    
    
    _harddrives = df.loc[3, :].tolist()
    for _harddrive in _harddrives[1:]:
        if _harddrive:
            _harddrive_id = get_harddrive_label(_harddrive)
            labels_index[_harddrive] = _harddrive_id
    
    #Graphics Coprocessor
    _graphprocessors = df.loc[4, :].tolist()
    for _graphprocessor in _graphprocessors[1:]:
        if _graphprocessor:
            _graphprocessor_id = get_graphprocessor_label(_graphprocessor)
            labels_index[_graphprocessor] = _graphprocessor_id
    
    
    _reviews = df.loc[5, :].tolist()
    for _t in _reviews[1:]:
        t =  " ".join(str(x) for x in _t)
        texts.append(t)
        #labels.append(_cpu_id)
        #labels.append(_sscreen_id)
        #labels.append(_ram_id)
        labels.append(_harddrive_id)
        #labels.append(_graphprocessor_id)
    """

    # get texts and labels
    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'

    #dir = '/data/raymond/workspace/exp2/'
    #file = dir +  'amazon_reviews.json'
    file = dir + 'amazon_reviews_copy.json'
    reviews = []
    texts, labels_lst = read_data_update.get_amazon_texts_labels(file)

    #
    labels_matrix = numpy.array(labels_lst)

    PERCENT = 0.1 # the range between the pred object
    _labels = labels_matrix[:, 0].tolist()
    _labels_PERCENT = labels_matrix[:, 0 : 7 * 1]
    
            
    #Found 838332 reviews
    print('Found %s reviews.' % len(texts))
    # define class labels
    #labels.ravel()
    #labels = array([1,1,1,1,1,0,0,0,0,0])
    #flattened_docs = [str(j) for i in docs for j in i]

    # prepare tokenizer
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    
    # integer encode the documents
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    #_word_index = dict((v, k) for k, v in word_index.items())
    #
    print('Found %s unique tokens.' % len(word_index))

    """
    # pad documents to a max length of 108 words
    # TODO - max_emb_encoder(x_emb, x_mask)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(numpy.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    """

    # load the whole glove embedding into memory
    embeddings_index = dict()
    f = open(dir + 'glove.6B.100d.txt',  "r", encoding="utf-8")
    for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    
    ## prepare embedding matrix
    print('Preparing embedding matrix.')

    # create a weight matrix for words in training docs
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    #embedding_matrix = numpy.random.uniform(-0.001, 0.001, (num_words, EMBEDDING_DIM))
    embedding_matrix = numpy.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be random in (-0.001, 0.001).
            embedding_matrix[i] = embedding_vector
         
    # pad documents to a max length of 108 words
    # TODO - max_emb_encoder(x_emb, x_mask) - padding the end
    # data (838332, 108)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    """
    indices = numpy.arange(data.shape[0]) 
    #_data = numpy.random.uniform(-0.001, 0.001, (data.shape))

    
    for i in indices: 
        # for each review

        # aver
        #numpy.average(data[i])

        #max_pooling - get the max index
        # (108, 100)
        max_matrix = numpy.random.uniform(-0.001, 0.001, (MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
        for j in range(len(data[i])):
            #word = _word_index[ind]
            ind = data[i,j]
            max_matrix[j] = embedding_matrix[ind]

        #data[i] = max_matrix.max(0)
        _data[i] = numpy.amax(max_matrix, axis=1)

        #index_max = numpy.argmax(data[i])
        #value_max = numpy.max(data[i])
        #data[i] = numpy.zeros((1,MAX_SEQUENCE_LENGTH))
        #data[i,index_max] = value_max    
    """

    labels = to_categorical(numpy.asarray(_labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int((VALIDATION_SPLIT + TEST_SPLIT) * data.shape[0])
    num_test_samples = int(TEST_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:-num_test_samples]
    y_val = labels[-num_validation_samples:-num_test_samples]

    x_test = data[-num_test_samples:]
    y_test = labels[-num_test_samples:]
    _y_test = _labels[-num_test_samples:]
    _y_test_PERCENT =_labels_PERCENT[-num_test_samples:]


    ## word embeddings
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    # embedding_layer (num_words, EMBEDDING_DIM)
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            #embeddings_initializer=Constant(embedding_matrix),
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False,
                            name="embedding_layer")

    print('Training model.')
    # define model

    # create model

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    #embedded_sequences = embedding_layer(sequence_input)

    def average_emb(input_seq):
        H_enc = tf.reduce_mean(input_seq, axis=1)  # batch 1 emb
        #x_sum = 
        #H_enc = K.mean(input_seq, axis=0) 
        #K.mean(x, axis=1, keepdims=True)
        #H_enc = np.amax(input_seq, axis=1)

        embedded_sequences = H_enc
        return embedded_sequences

    def max_emb(input_seq):
        H_enc = tf.reduce_max(input_seq, axis=1)  # batch 1 emb 
        #H_enc = K.max(input_seq, axis=1)
        #H_enc = np.amax(input_seq, axis=1)

        embedded_sequences = H_enc
        return embedded_sequences

    def concat_emb(input_seq):
        H_enc_1 = max_emb(input_seq)
        H_enc_2 = average_emb(input_seq)

        embedded_sequences = tf.concat([H_enc_2, H_enc_1], 1)
        #embedded_sequences = K.concatenate([H_enc_2, H_enc_1], 1)
        return embedded_sequences

    embedded_sequences = embedding_layer(sequence_input) # (batch , 54, 100)

    #max_emb_encoder(sequence_input, )
    #xx = Lambda(average_emb)(embedded_sequences)  #(?, 100)
    #xx = Lambda(max_emb)(embedded_sequences)  #(?, 100)
    xx = Lambda(concat_emb)(embedded_sequences)

    #xx = Dense(EMBEDDING_DIM, activation='relu')(xx)
    #xx = Flatten()(embedded_sequences)
    #model.add(Dense(labels.shape[1], activation='softmax'))
    #xx = Flatten()(xx)
    preds = Dense(labels.shape[1], activation='softmax')(xx)

    """   
    # train a 1D convnet with global maxpooling
    #sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    #embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 3, activation='relu')(embedded_sequences)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(labels.shape[1], activation='softmax')(x)
    """
    
    model = Model(sequence_input, preds)

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', keras_metrics.precision(), keras_metrics.recall()])

    # summarize the model
    print(model.summary())

    # fit the model
    hist = model.fit(x_train, y_train,
          batch_size=50,
          epochs=5,
          validation_data=(x_val, y_val))

    print(hist.history)

    # evaluate the model
    #print('recall + precision: %s' % (_metrics.get_data() * 100))
    
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    print('accuracy: %f, precision: %f, recall: %f' % (accuracy, precision, recall))
    #print('precision: %f' % precision)
    #print('recall: %f' % recall)

    #predict
    y_proba = model.predict(x_test)
    y_proba_ind = numpy.argsort(-y_proba)
    print("y_proba: ", y_proba_ind)

    K = 40

    #num = 0
    #y_test = y_lst[:, 0]
    y_pred_max = y_proba_ind[:, 0]

    #precision_1 = metrics.precision_score(_y_test, y_pred_max, average='macro')
    #print("precision_1: %f" % precision_1)
        
    #num_lst = []
 
    #num = numpy.sum(y_test == y_pred_max)
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

        recall = _recall_score(y_pred, _y_test)
        recalls.append(recall)
        print(recall)

    """
    print(scores.keys())
    print(scores['test_precision_macro'])
    print(scores['test_recall_macro'])

    """
    print(numpy.mean([precisions, recalls], 0))
    #print(f1)
   
    return 0


def accuracy_score(y_test, y_pred):
    accuracy = 0

    """
    # Number of relevant and recommended items in top k
    n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

    # Precision@K: Proportion of recommended items that are relevant
    precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

    # Recall@K: Proportion of relevant items that are recommended
    recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    """

    num = 0
    i = 0
    while i < len(y_test):
        lst = y_test[i].tolist()
        if y_pred[i] in lst:
            num = num + 1

        i = i + 1
    
    accuracy = num / len(y_pred)

    return accuracy


def train_svm():
        # get documents
    """
    f1 = open('C:/Users/raymondzhao/myproject/dev.dplearning/data/amazon_data_0719.p', 'r')
    asins = pickle.load(f1)
    f1.close()
    """
    """
    #dir = 'C:/Users/raymondzhao/myproject/dev.dplearning/data/'
    #dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    dir = "/data/raymond/workspace/exp2/"
    file = dir + 'amazon_reviews.json'
    asins = _read_data(file)

    #file = dir + 'amazon_tech_all_5.csv'
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
            labels.append(_cpu_id)
            #labels.append(_sscreen_id)
            #labels.append(_ram_id)
            #labels.append(_harddrive_id)
            #labels.append(_graphprocessor_id)
    """

    """
    _cpus = df.loc[1, :].tolist()
    for _cpu in _cpus[1:]:
        #_cpu = asins[_asin][1]
        if _cpu:
           _cpu_id = get_cpu_label(_cpu)
           labels_index[_cpu] = _cpu_id

    _sscreens = df.loc[0, :].tolist()
    for _sscreen in _sscreens[1:]:
        #_sscreen = asins[_asin][0]
        if _sscreen:
            _sscreen_id = get_sscreen_label(_sscreen)
            labels_index[_sscreen] = _sscreen_id
    
    _rams = df.loc[2, :].tolist()
    for _ram in _rams[1:]:
        if _ram:
            _ram_id = get_ram_label(_ram)
            labels_index[_ram] = _ram_id
    
    
    _harddrives = df.loc[3, :].tolist()
    for _harddrive in _harddrives[1:]:
        if _harddrive:
            _harddrive_id = get_harddrive_label(_harddrive)
            labels_index[_harddrive] = _harddrive_id
    
    #Graphics Coprocessor
    _graphprocessors = df.loc[4, :].tolist()
    for _graphprocessor in _graphprocessors[1:]:
        if _graphprocessor:
            _graphprocessor_id = get_graphprocessor_label(_graphprocessor)
            labels_index[_graphprocessor] = _graphprocessor_id
    
    
    _reviews = df.loc[5, :].tolist()
    for _t in _reviews[1:]:
        t =  " ".join(str(x) for x in _t)
        texts.append(t)
        #labels.append(_cpu_id)
        #labels.append(_sscreen_id)
        #labels.append(_ram_id)
        labels.append(_harddrive_id)
        #labels.append(_graphprocessor_id)
    """

    # get texts and labels
    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'

    #dir = '/data/raymond/workspace/exp2/'
    #file = dir +  'amazon_reviews.json'
    file = dir + 'amazon_reviews_copy.json'
    reviews = []
    asins_dict = read_data_update.get_amazon_texts_labels(file)
    texts = []
    labels_lst = []
    for asin in asins_dict:
        texts.append(asins_dict[asin][0])
        labels_lst.append(asins_dict[asin][1])

    #
    labels_matrix = numpy.array(labels_lst)
    labels = labels_matrix[:, 0].tolist()
    
    #Found  reviews
    print('Found %s reviews.' % len(texts))

    # prepare tokenizer
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    
    # integer encode the documents
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    #_word_index = dict((v, k) for k, v in word_index.items())
    #
    print('Found %s unique tokens.' % len(word_index))

    """
    # pad documents to a max length of 108 words
    # TODO - max_emb_encoder(x_emb, x_mask)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(numpy.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    """

    # load the whole glove embedding into memory
    embeddings_index = dict()
    f = open(dir + 'glove.6B.100d.txt',  "r", encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    
    # prepare embedding matrix
    print('Preparing embedding matrix.')

    # create a weight matrix for words in training docs
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = numpy.random.uniform(-0.001, 0.001, (num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be random in (-0.001, 0.001).
            embedding_matrix[i] = embedding_vector
         
    # pad documents to a max length of 108 words
    # TODO - max_emb_encoder(x_emb, x_mask) - padding the end
    # data (838332, 108)
    #data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    data = numpy.random.uniform(-0.001, 0.001, (len(sequences), EMBEDDING_DIM))
    indices = numpy.arange(data.shape[0]) 
    #review_feature_vecs = numpy.zeros((len(reviews), num_features), dtype="float32")
    
    for seq in sequences:
        #for each review
        seq_ind = 0
        word_cnt = 0
        feature_vec = numpy.random.uniform(-0.001, 0.001, (EMBEDDING_DIM))
        for word_index in seq:
            word_cnt += 1
            # ind = seq[word_index]
            feature_vec += embedding_matrix[word_index]

        # mean
        feature_vec /= word_cnt

        # max
        data[seq_ind] = feature_vec
        seq_ind += 1

    """
    for i in indices: 
        # for each review

        # aver
        #numpy.average(data[i])

        #max_pooling - get the max index
        # (108, 100)
        max_matrix = numpy.random.uniform(-0.001, 0.001, (MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
        for j in range(len(data[i])):
            #word = _word_index[ind]
            ind = data[i,j]
            max_matrix[j] = embedding_matrix[ind]

        #data[i] = max_matrix.max(0)
        _data[i] = numpy.amax(max_matrix, axis=1)

        #index_max = numpy.argmax(data[i])
        #value_max = numpy.max(data[i])
        #data[i] = numpy.zeros((1,MAX_SEQUENCE_LENGTH))
        #data[i,index_max] = value_max    
        """
    

    #labels = to_categorical(numpy.asarray(labels))
    print('Shape of data tensor:', data.shape)
    #print('Shape of label tensor:', labels.shape)

    #it the data into a training set and a validation set
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    #labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    num_test_samples = int(0.2 * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_test = data[-num_test_samples:]
    y_test_1 = labels[-num_test_samples:]
    x_val = data[-num_validation_samples:]
    y_val_1 = labels[-num_validation_samples:]

    y_test_k = labels_matrix[ (len(labels_matrix)  - num_test_samples) : (len(labels_matrix) + 1), : ]

    #
    #C = 1.0  # SVM regularization parameter
    print('Classify ... ')
    # svm_classifier = svm.SVC(kernel='linear', gamma=2, C=1.0)
    svm_classifier = svm.SVC(kernel='rbf', gamma=2, C=1.0) #Gaussian Kernel
    svm_classifier.fit(x_train,y_train)

    y_pred = svm_classifier.predict(x_test)
    
    K = 70

    num = 0
    y_test = y_test_k[:, 1]
    num = numpy.sum(y_test == y_pred)

    acc1 = num/len(y_pred)
    print(acc1)

    i = 1
    while i < K:
        i = i+1
        y_test = y_test_k[:, 0:i]

        #print(confusion_matrix(y_test, y_pred))

        print(accuracy_score(y_test, y_pred))
        #print(recall_score(y_test, y_pred))
        #print(f1_score(y_test, y_pred))

        #print(classification_report(y_test, y_pred))

    return 0

def main():
    #data()
    #
    #train_wordembedding()
    #
    train_svm()

    #train_mlp()


if __name__ == '__main__':
    main()
