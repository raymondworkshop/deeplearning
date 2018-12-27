"""
classify laptop using SVM
@author Raymond

"""

import pandas as pd
import numpy as np
#from pandas import ExcelFile
#from pandas import ExcelWriter

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
#from sklearn.metrics import recall_score
#from sklearn.metrics.scorer import make_scorer
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix


#from sklearn.model_selection import cross_val_score

values = {'cpu': 1, 'RAM':3, 'HDD':3, 'monitor':2, 'graphical':5}
asin_map_price = {1:10999, 2:7799, 3:5399, 4:6999, 5:9999, 6:6799, 7:8199, 8:9099, 9:7599, 10:12999, 11:3599}
price_map_asin = {10999:1, 7799:2, 5399:3, 6999:4, 9999:5, 6799:6, 8199:7, 9099:8, 7599:9, 12999:10, 3599:11}

""" map
2.16 GHz Intel Celeron : 1
2.2 GHz Intel Core i5  : 2
2.5 GHz Intel Core i5  : 3
3 GHz 8032             : 4
1.6 GHz                 
2 GHz AMD A Series
2.5 GHz Pentium
3.5 GHz 8032 
2.7 GHz Intel Core i7
1.6 GHz Intel Celeron 
2.8 GHz Intel Core i7
2 GHz AMD A Series
2.8 GHz Intel Core i7 
2.5 GHz AMD A Series 
1.6 GHz Intel Core i5
3.8 GHz Core i7 Family
2.5 GHz Intel Core i5
2.7 GHz Intel Core i7
1.8 GHz Intel Core i7 
1.1 GHz Pentium
3 GHz AMD A Series 
2.5 GHz Intel Core i5 
2.3 GHz Intel Core i5 
Intel Core i5
2.4 GHz Intel Core i3
2.4 GHz Intel Core i3 
3.1 GHz Intel Core i5
3.8 GHz Intel Core i7 
1.6 GHz Intel Celeron 
4 GHz Intel Core i7 
1.6 GHz Intel Core 2 Du
2.9 GHz Intel Celeron 
3.6 GHz AMD A Series 
1.1 GHz Intel Celeron
1.6 GHz Intel Celeron
2 GHz 
2.4 GHz Intel Core i3
1.6 GHz Intel Core i5
2.5 GHz Intel Core i5 
2.4 GHz Intel Core i3 
2.48 GHz Intel Celeron
1.6 GHz Intel Celeron
1.6 GHz Intel Celeron
2.2 GHz Intel Core i3 
2.3 GHz Intel Core i3 
Intel 4 Core 
1.6 GHz AMD E Series
3.5 GHz 8032
3.5 GHz Intel Core i5 
2.4 GHz Intel Core i3
2.7 GHz AMD A Series
2.7 GHz Intel Core i7 
2.5 GHz Intel Core i5 
2.5 GHz Core i5 7200U
1.6 GHz Intel Mobile CP
1.6 GHz Intel Celeron 
2.8 GHz Intel Core i7
1.8 GHz Intel Core i7
Celeron N3060 
3 GHz AMD A Series
1.6 GHz Intel Celeron 
2.5 GHz Intel Core i5 
2.1 GHz Intel Core i7 
1.1 GHz Intel Celeron 
2.7 GHz Core i7 7500U 
1.8 GHz AMD E Series
1.5 GHz
1.7 GHz Exynos 5000 Ser
1.8 GHz 8032 
2 GHz AMD A Series
8032
2.7 GHz Core i7 2.7 GHz
1.7 GHz
3.8 GHz Intel Core i7
1.6 GHz Intel Core i5
2.5 GHz Intel Core i5
2.16 GHz Athlon 2650e
2.3 GHz Intel Core i5
2.5 GHz Intel Core i5
2.5 GHz Pentium
2.4 GHz Intel Core i3
1.6 GHz Celeron N3050
3.4 GHz Intel Core i5
3.5 GHz Intel Core i5
2.7 GHz AMD A Series
3.5 GHz Intel Core i7
2.5 GHz Intel Core i5
3 GHz
2.4 GHz Core i3-540
2.8 GHz 8032
2.7 GHz Intel Core i3 
2.6 GHz Intel Core i5 
1.1 GHz Pentium
3.4 GHz Intel Core i5 
3.4 GHz Intel Core i5 
2.8 GHz Intel Core i7 
2.5 GHz Intel Core i5 
1.6 GHz
2.7 GHz 8032
2.5 GHz Athlon 2650e
1.8 GHz Intel Core i7 
2.4 GHz Intel Core i3
2.5 GHz Intel Core Duo
1.6 GHz Celeron N3060 
2.7 GHz Intel Core i7 
1.1 GHz Intel Celeron 
2.5 GHz Intel Core i5 
2.4 GHz AMD A Series
1.6 GHz Intel Celeron
2.3 GHz Intel Core i5
2.7 GHz Intel Core i7
1.1 GHz Intel Celeron
2 GHz Celeron D Process
1.6 GHz Intel Core i5
2.4 GHz AMD A Series
2.16 GHz Intel Celeron
"""

def get_inds(item, lst):
    """
    get the list according to the min-distance with item
    """
    # amazon_refurbished_goods_index
    _lst = [ abs(x - item) for x in lst ]

    #ind = _lst.index(min(_lst))
    prices = [x for _, x in sorted(zip(_lst, lst))]

    inds = []
    for price in prices:
        inds.append(price_map_asin[price])


    return inds


def sort_items():
    # sort item 

    price_lst = []
    #label_lst = []
    #asin_map_price = {}

    # map params to prices 
    # get amazon_refurbished_goods_index
    _dict = {} # sort according to the price

    for asin in asin_map_price:
        price = asin_map_price[asin]
        price_lst.append(price)
        #_dict[asin] = get_inds(price, price_lst) 
    #
    for asin in asin_map_price:
        price = asin_map_price[asin]
        #price_lst.append(price)
        _dict[asin] = get_inds(price, price_lst) 
    #
    return _dict


def _precision_score(y_test, y_pred):
    precision = 0

    num = 0
    i = 0
    while i < len(y_test):
        lst = y_test[i].tolist()
        if y_pred[i] in lst:
            num = num + 1

        i = i + 1
    
    precision = num / (len(y_pred) * y_test.shape[1])
    
    
    """
    i = set(y_test).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1
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


def train(file):
    # loading the dataset
    df = pd.read_excel(file)
    df.head()
    df.info()

    #replaced the missing data with the mean of that column
    #df.fillna(df.mean(), inplace=True)
    #df.fillna(df.mean(), inplace=True)
    df.fillna(value=values, inplace=True)

    # X - > features, y->label
    y = df['class label']
    y_price = []
    for _y in y:
        y_price.append(asin_map_price[_y])

    _dict = sort_items()

    _y_lst = []
    for _y in y:
        print(_dict[_y])
        _y_lst.append(_dict[_y])

    # lst according to inds
    y_lst = np.array(_y_lst)

    X = df.iloc[:, [1, 2, 3, 4, 5]]

    #X_cpu = df['cpu']
    #X_ram = df['RAM']
    #X_hdd = df['HDD']
    #X_monitor = df['monitor']
    #X_graphical = df['graphical']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    """
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)
    acc = svm_model_linear.score(X_test, y_test)
    cm = confusion_matrix(y_test, svm_predictions)
    print(acc)
    print(cm)
    """
    classifier = OneVsRestClassifier(SVC(kernel='linear', C=1, probability=True, random_state=0))
    #scoring = ['precision_macro', 'recall_macro']
    y_pred = cross_val_predict(classifier, X, y, cv=10)
    _y_proba = cross_val_predict(classifier, X, y, cv=10, method='predict_proba')
    #classifier.fit(X_train, y_train)
    #y_pred = classifier.predict(X_test)
    #_y_proba = classifier.predict_proba(X_test)
    y_proba = np.argsort(-_y_proba,axis=1) + 1
    
    K = 11

    #num = 0
    y_test = y_lst[:, 0]
    y_pred_max = y_proba[:, 0]

    precision_1 = metrics.precision_score(y_test, y_pred_max, average='macro')
    print("precision_1: %f" % precision_1)
        
    #num_lst = []
 
    num = np.sum(y_test == y_pred_max)
    #num_lst.append(num)

    i = 1
    while i < K:
        #y_test = y_lst[:, 0:i+1]
        y_pred = y_proba[:, 0:i+1]

        i = i+1

        #precision, recall, fscore, support = score(y_test, y_pred_max)
        #precision_n = metrics.precision_score(y_test, y_pred, average='macro')
        #print("precision_n: %f" % precision_n)

        #_num = np.sum(y_test == y_pred)
        #num_lst.append(_num)
        #num = _num + num

        #recall = num/len(y_pred)
        #print(recall)

        #precision = num/len(y_pred)
        #print(precision)

        #print(confusion_matrix(y_test, y_pred))

        print(_precision_score(y_pred, y_test))
        #print(_recall_score(y_pred, y_test))
        #print(f1_score(y_test, y_pred))

        #print(classification_report(y_test, y_pred))

    print("recall:")
    #recall = []
    recall_1 = metrics.recall_score(y_test, y_pred_max, average='micro')
    #recall.append(recall_1)
    print("recall_1: %f" % recall_1)
    
    i = 1
    while i < K:
        #y_test = y_lst[:, 0:i+1]
        y_pred = y_proba[:, 0:i+1]

        i = i+1
        print(_recall_score(y_pred, y_test))

    """
    print(scores.keys())
    print(scores['test_precision_macro'])
    print(scores['test_recall_macro'])

    """

    return 0

#
def main():
    #dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    file = "C:\\Users\\raymondzhao\\myproject\\dev.deeplearning\\data\\laptop_data_new.xlsx"
    #file = "C:\\Users\\raymondzhao\\myproject\\data\\laptop_data_new.xlsx" 

    train(file)

    print("Done")

#
if __name__ == '__main__':
    main()