import os
import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import MaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import math
import read_data_update


# the dataset path
#TEXT_DATA_DIR = r'../input/20-newsgroup-original/20_newsgroup/20_newsgroup/'
TEXT_DATA_DIR = "C:/Users/raymondzhao/myproject/dev.deeplearning/data/20_newsgroup/"
#the path for Glove embeddings
#GLOVE_DIR = "C:/Users/raymondzhao/myproject/dev.deeplearning/data/glove.6B"

#GLOVE_DIR = r'../input/glove6b/'
# make the max word length to be constant
MAX_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
# the percentage of train test split to be applied
VALIDATION_SPLIT = 0.20
TEST_SPLIT = 0.1
# the dimension of vectors to be used
EMBEDDING_DIM = 100
# filter sizes of the different conv layers 
filter_sizes = [3,4,5]
num_filters = 512
embedding_dim = 100
# dropout probability
drop = 0.5
batch_size = 30
epochs = 2


## preparing dataset
"""
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)
print(labels_index)
"""

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

def train_CNN():
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

texts, labels_matrix = train_CNN()
_labels = labels_matrix[:, 0].tolist()
_labels_PERCENT = labels_matrix[:, 0 : math.ceil(labels_matrix.shape[1] * 0.2)]
  

print('Found %s texts.' % len(texts))

tokenizer  = Tokenizer(num_words = MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences =  tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("unique words : {}".format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#labels = to_categorical(np.asarray(labels))
labels = to_categorical(np.asarray(_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print(labels)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
#nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
nb_validation_samples = int((VALIDATION_SPLIT + TEST_SPLIT) * data.shape[0])

nb_test_samples = int(TEST_SPLIT * data.shape[0])


x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_test_samples:]
y_test = labels[-nb_test_samples:]
_y_test = _labels[-nb_test_samples:]
_y_test_PERCENT =_labels_PERCENT[-nb_test_samples:]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


GLOVE_DIR = "C:/Users/raymondzhao/myproject/dev.deeplearning/data/"
embeddings_index = {}
f = open(GLOVE_DIR + 'glove.6B.100d.txt',  "r", encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding = embedding_layer(inputs)

print(embedding.shape)
reshape = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(embedding)
print(reshape.shape)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
#output = Dense(units=20, activation='softmax')(dropout)
output = Dense(labels.shape[1], activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights_cnn_sentece.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("Traning Model...")
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(x_val, y_val))


#bsize = 64
#epoch = int(len(x_train) / bsize)
#epoch = 12
for epoch in range(20):
    hist = model.fit(x_train, y_train, 
                     verbose=1, 
                     #validation_split=VALIDATION_SPLIT,
                     validation_data=(x_val, y_val),
                     epochs=epoch,
                     callbacks=[checkpoint],
                     batch_size=batch_size)
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