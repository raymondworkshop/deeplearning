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


import pickle
import re
import ast
import math

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 300  #300
EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.3

def get_cpu_label(_str):
    # [ 2 GHz AMD A Series, 1.1 GHz Intel Celeron, 2.16 GHz Intel Celeron,3 GHz 8032,1.6 GHz,3.5 GHz 8032,4 GHz Intel Core i7]
    _cpu_map = {
        "amd": 0,
        "1.1 Intel":1,
        "2.0-2.9 Intel": 2,
        "3.0-3.9 Intel":3,
        "4 Intel":4,
        "others":5 
    }

    _cpu_label = 5 #unknown
    if 'AMD|amd' in _str:
        _cpu_label = 0
    else: #Intel
        _cpu_frequency = round(float(re.search('[\d]+[.\d]*', _str).group()))
        if _cpu_frequency == 1:
            _cpu_label = 1
        elif _cpu_frequency == 2:
            _cpu_label = 2
        elif _cpu_frequency == 3:
            _cpu_label = 3
        elif _cpu_frequency == 4:
            _cpu_label = 4
        else:
            pass

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
    if 'inches' in _str:
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
        "8 GB SDRAM DDR4": 4,
        "12 GB DDR SDRAM":5,
        "16 GB DDR4":6
    }

    _ram_label = 7 #unknown
    if 'GB'  in _str:
        _ram_size = int(float(re.search('[\d]+[.\d]*', _str).group()))
        if _ram_size == 2:
            _ram_label = 0
        elif _ram_size == 4:
            _ram_label = 1
        elif _ram_size  == 6:
            _ram_label = 2
        elif _ram_size == 8:
            if 'DDR3' in _str:
                _ram_label = 3
            if 'DDR4' in _str:
                _ram_label = 4
        elif _ram_size  == 12:
            _ram_label = 5
        elif _ram_size  == 16:
            _ram_label = 6
        else:
            _ram_label = 7

    return _ram_label
    
    return _ram_label

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
                        if len(asins[_asin]) == 3:
                            break
                    asins[_asin].append(list_reviews)

    return asins

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

    # The text samples and their labels
    texts = []  #list of text samples
    #labels = array([])
    labels = [] #list of label ids
    labels_index = {}  # dictionary mapping label name to numeric id

    # ['14 inches', '2.16 GHz Intel Celeron', '4 GB SDRAM DDR3', 
    # [[b'I', b'placed', b'my', b'order', b'on', b'December', b'19th', b'and', b'was', b'under', b'the', b'impression', b'it', b'would', b'arrive', b'on', b'the', b'22nd']]
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
        
        #reviews
        reviews = asins[_asin][3] 
        for _t in reviews:
            t =  " ".join(str(x) for x in _t)
            texts.append(t)
            #labels.append(_cpu_id)
            #labels.append(_sscreen_id)
            labels.append(_ram_id)
            

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
    #
    print('Found %s unique tokens.' % len(word_index))

    # pad documents to a max length of 300 words
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
    embedding_matrix = numpy.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            #embeddings_initializer=Constant(embedding_matrix),
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

    print('Training model.')
    # define model

    # create model
    """
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    # a 1D convolutional neural network (CNN)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(6, activation='softmax')(x)

    """
    """
    x = Dense(1, activation='relu',input_dim=300)(embedded_sequences)
    preds = Dense(6, activation='softmax')(x)
    """
    #vsize = 100
    #vv = ThetaPrime(vsize)

    model = Sequential()

    def average_emb(input_seq):
        # the mean
        H_enc = tf.reduce_mean(input_seq, axis=1)  # batch 1 emb
        #H_enc = tf.reduce_max(input_seq, axis=1)

        embedded_sequences = H_enc
        return embedded_sequences

    #input_sequences = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    #xx = embedding_layer(input_sequences)
    #xx = Lambda(average_emb)(xx)

    #preds = Dense(8, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0), activation='sigmoid')(xx)
    #preds = Dense(8, activation='sigmoid')(xx)
    #model.add(Dense(8, activation='sigmoid')(xx))
    #model = Model(input_sequences, preds)
    #model = ExtendedModel(input=input_sequences, output=preds)

    #model.add_extra_trainable_weight(tf.Variable(numpy.zeros((vsize), dtype='float32')))
    #model = Model(input_sequences, preds)
    
    """
    model.add(xx)
    model.add(Flatten())
    model.add(Dense(6, activation='sigmoid'))
    """

    
    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(8, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          validation_split=.1)

    # evaluate the model
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

    return 0

def main():
    #data()
    #
    train()

if __name__ == '__main__':
    main()
