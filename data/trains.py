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

from engine_training import ExtendedModel
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

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 54 * 2 #300
EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.1


def get_cpu_label(_str):
    # [ 2 GHz AMD A Series, 1.1 GHz Intel Celeron, 2.16 GHz Intel Celeron,3 GHz 8032,1.6 GHz,3.5 GHz 8032,4 GHz Intel Core i7]
    _cpu_map = {
        "amd": 0,
        "1.1 Intel":1,
        "1.5-2.5 Intel": 2,
        "2.6-3.5 Intel":3,
        "3.6 - 4 Intel":4,
        "others":4
         
    }

    _cpu_label = 4 #unknown
    if 'amd' in _str.lower():
        _cpu_label = 0
    if 'intel' in _str.lower(): #Intel
        _cpu_frequency = int(float(re.search('[\d]+[.\d]*', _str).group()))
        if _cpu_frequency == 1:
            _cpu_label = 1
        elif _cpu_frequency == 2:
            _cpu_label = 2
        elif _cpu_frequency == 3:
            _cpu_label = 3
        elif _cpu_frequency == 4:
            _cpu_label = 4
        else:
            _cpu_label = 4

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
        "2 GB SDRAM": 1,
        "4 GB SDRAM DDR3": 1,
        "6 GB DDR SDRAM":2,
        "8 GB SDRAM DDR3": 2,
        "8 GB SDRAM DDR4": 2,
        "12 GB DDR SDRAM":3,
        "16 GB DDR4" :3,
        "others":4,
    }

    _ram_label = 4 #unknown
    if 'gb'  in _str.lower():
        _ram_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
        if _ram_size == 2:
            _ram_label = 1
        elif _ram_size == 4:
            _ram_label = 1
        elif _ram_size  == 6:
            _ram_label = 2
        elif _ram_size == 8:
            if 'ddr3' in _str.lower():
                _ram_label = 2
            elif 'ddr4' in _str.lower():
                _ram_label = 2
            else:
                _ram_label = 2
        elif _ram_size  == 12:
            _ram_label = 3
        elif _ram_size  == 16:
            _ram_label = 3
        else:
            _ram_label = 4

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
        "HDD ~= 1T" :2,
        "HDD ~= 500G" :4,
        "HDD < 500G" :4,
        "others": 6
    }

    _harddrive_label = 6 #unknown
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
                    _harddrive_label = 2
                else:
                    _harddrive_label = 2
            else:
                _harddrive_label = 2
        else:
            if num_there(_str):
                _harddrive_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
                if _harddrive_size >= 500:
                    _harddrive_label = 4
                else:
                    _harddrive_label = 4
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
        "AMD Radeon R2": 2,
        "AMD Radeon R5": 3,
        "AMD Radeon R7": 3,
        "AMD Radeon R4" :2,
        "NVIDIA GeForce GTX 1050": 4,
        "NVIDIA GeForce 940MX" :  4,
        "Integrated" : 5,
        "others| PC | FirePro W4190M ": 6
    }

    _graphprocessor_label = 6 #unknown
    if 'intel' in _str.lower():
        if num_there(_str):
            _graphprocessor_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
            if _graphprocessor_size == 500:
                _graphprocessor_label = 0
            elif _graphprocessor_size == 505:
                _graphprocessor_label = 0
            elif _graphprocessor_size  == 620:
                _graphprocessor_label = 1  
            else:
                _graphprocessor_label = 1
        else:
            _graphprocessor_label = 1

    if 'amd' in _str.lower():
        if 'r2' in _str.lower():
            _graphprocessor_label = 2
        if 'r5' in _str.lower():
            _graphprocessor_label = 3
        if 'r7' in _str.lower():
            _graphprocessor_label = 3
        if 'r4' in _str.lower():
            _graphprocessor_label = 2

    if 'nvidia' in _str.lower():
        if num_there(_str):
            _graphprocessor_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
            if _graphprocessor_size == 1050:
                _graphprocessor_label = 4
            if _graphprocessor_size == 940:
                _graphprocessor_label = 4

    if 'integrated' in _str.lower():
        _graphprocessor_label  = 5
    

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

                    asins[_asin].append(list_reviews)

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


def train():
    # get documents
    """
    f1 = open('C:/Users/raymondzhao/myproject/dev.dplearning/data/amazon_data_0719.p', 'r')
    asins = pickle.load(f1)
    f1.close()
    """
    dir = "C:/Users/raymondzhao/myproject/dev.dplearning/data/"
    #dir = "/data/raymond/workspace/exp2/"
    file = dir + 'amazon_reviews.json'
    asins = read_data(file)

    file = dir + 'amazon_tech_all_5.csv'
    #df = pd.read_csv(file)
    
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

    #'14 inches', '2.16 GHz Intel Celeron', '4 GB SDRAM DDR3', '16 GB SSD', 'Intel HD Graphics', 'Intel HD Graphics',

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
            #stripped = [w.decode("utf-8").translate(table) for w in _t] 
            #stripped = [w.decode("utf-8").lower().translate(table) for w in _t] 
            s = " ".join(x for x in stripped)

            #t = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+','', t)
            texts.append(s)
            labels.append(_cpu_id)
            #labels.append(_sscreen_id)
            #labels.append(_ram_id)
            #labels.append(_harddrive_id)
            #labels.append(_graphprocessor_id)
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
    
    #sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    #embedded_sequences = embedding_layer(sequence_input)
    
    """
    x = Conv1D(128, 3, activation='relu')(embedded_sequences)
    
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(7, activation='softmax')(x)
    #x = Dense(1, activation='relu')(embedded_sequences)
    #preds = Dense(6, activation='softmax')(x)
    """
    

    """
    def average_emb(input_seq):
        # the mean
        H_enc = tf.reduce_mean(input_seq, axis=1)  # batch 1 emb
        #H_enc = tf.reduce_max(input_seq, axis=1)

        embedded_sequences = H_enc
        return embedded_sequences
    """

    #input_sequences = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    #xx = embedding_layer(input_sequences)
    #xx = Lambda(average_emb)(xx)

    #preds = Dense(8, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0), activation='sigmoid')(xx)
    #preds = Dense(8, activation='sigmoid')(xx)
    #model.add(Dense(8, activation='sigmoid')(xx))
    #model = Model(input_sequences, preds)
    #model = ExtendedModel(input=input_sequences, output=preds)

    #model.add_extra_trainable_weight(tf.Variable(numpy.zeros((vsize), dtype='float32')))
    #model = Model(sequence_input, preds)
    
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    #embedded_sequences = embedding_layer(sequence_input)

    #model = Sequential()
    #model.add(embedding_layer)

    def max_emb(input_seq):

        H_enc = tf.reduce_max(input_seq, axis=1)  # batch 1 emb
        #H_enc = np.amax(input_seq, axis=1)

        embedded_sequences = H_enc
        return embedded_sequences
    
    def average_emb(input_seq):
        H_enc = tf.reduce_mean(input_seq, axis=1)  # batch 1 emb
        #H_enc = np.amax(input_seq, axis=1)

        embedded_sequences = H_enc
        return embedded_sequences

    def concat_emb(input_seq):
        H_enc_1 = max_emb(input_seq)
        H_enc_2 = average_emb(input_seq)

        embedded_sequences = tf.concat([H_enc_2, H_enc_1], 1)
        return embedded_sequences

    embedded_sequences = embedding_layer(sequence_input) # (batch , 54, 100)

    #max_emb_encoder(sequence_input, )
    xx = Lambda(average_emb)(embedded_sequences)  #(?, 100)
    ##xx = Lambda(max_emb)(embedded_sequences)  #(?, 100)
    #xx = Lambda(concat_emb)(embedded_sequences)

    #xx = Dense(EMBEDDING_DIM, activation='relu')(xx)
    ##xx = Flatten()(embedded_sequences)
    #model.add(Dense(labels.shape[1], activation='softmax'))
    preds = Dense(labels.shape[1], activation='softmax')(xx)

    model = Model(sequence_input, preds)

    """
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)
    """

    #_precision = keras_metrics.precision()
    #_recall = keras_metrics.recall()
    #_metrics = Metrics()

    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    embedded_sequences = embedding_layer(sequence_input)
    preds = Dense(labels.shape[1], activation='softmax')(embedded_sequences)

    model = Model(sequence_input, preds)
    """

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', keras_metrics.precision(), keras_metrics.recall()])

    # summarize the model
    print(model.summary())

    # fit the model
    hist = model.fit(x_train, y_train,
          batch_size=128,
          epochs=8,
          validation_data=(x_val, y_val))

    print(hist.history)

    # evaluate the model
    #print('recall + precision: %s' % (_metrics.get_data() * 100))
    
    loss, accuracy, precision, recall = model.evaluate(x_val, y_val, verbose=0)
    print('metrics: %f' % accuracy)

    return 0

def main():
    #data()
    #
    train()

if __name__ == '__main__':
    main()
