# based on GloVe word embeddings
# @raymond
#
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

import pickle
import re
import ast

def get_cpu_label(_str):
    # [ 2 GHz AMD A Series, 1.1 GHz Intel Celeron, 2.16 GHz Intel Celeron,3 GHz 8032,1.6 GHz,3.5 GHz 8032,4 GHz Intel Core i7]
    _cpu_map = {
        "amd": 0,
        "1.1 Intel":1,
        "2.0-2.9 Intel": 2,
        "3.0-3.9 Intel":3,
        "4 Intel":4
    }
    _cpu = re.search('AMD|amd', _str).group()
    if _cpu: #AMD
        _cpu_label = 0 
    else: #Intel
        _cpu_frequency = re.search('[\d]+[.\d]*', _str).group()
        if int(_cpu_frequency) == 1:
            _cpu_label = 1
        elif int(_cpu_frequency) == 2:
            _cpu_label = 2
        elif int(_cpu_frequency) == 3:
            _cpu_label = 3
        elif int(_cpu_frequency) == 4:
            _cpu_label = 4
        else:
            pass

    return _cpu_label

def get_sscreen_label(_str):

    return

def get_ram_label(_str):

    return

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

def data():
    # get documents
    """
    f1 = open('C:/Users/raymondzhao/myproject/dev.dplearning/data/amazon_data_0719.p', 'r')
    asins = pickle.load(f1)
    f1.close()
    """
    file = 'C:/Users/raymondzhao/myproject/dev.dplearning/data/amazon_reviews.json'
    asins = read_data(file)

    docs = []
    labels = array([])
    for _asin in asins:
        print("The asin %s:", _asin)
        # [screensize,cpu, ram, reviews]
        _cpu = asins[_asin][1]
        if _cpu:
           _cpu_label = get_cpu_label(_cpu)
        #reviews
        _reviews = asins[_asin][3] 
        docs = docs + _reviews

        _labels = array([_cpu_label] * len(_reviews))
        labels.append(_labels)

    # define class labels
    labels.ravel()
    #labels = array([1,1,1,1,1,0,0,0,0,0])

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print(encoded_docs)
    # pad documents to a max length of 4 words
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)

    return

def train():
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('C:/Users/raymondzhao/myproject/dev.dplearning/data/glove.6B.100d.txt',  "r", encoding="utf-8")
    for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    """
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
		    embedding_matrix[i] = embedding_vector
    # define model
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    """

    return

def main():
    data()
    #
    #train()

if __name__ == '__main__':
    main()
