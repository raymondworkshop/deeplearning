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
from sklearn.metrics import recall_score
#from sklearn.metrics.scorer import make_scorer

#from sklearn.model_selection import cross_val_score

values = {'cpu': 1, 'RAM':3, 'HDD':3, 'monitor':2, 'graphical':5}
asin_map_price = {1:10999, 2:7799, 3:5399, 4:6999, 5:9999, 6:6799, 7:8199, 8:9099, 9:7599, 10:12999, 11:3599}

def get_inds(item, lst):
    """
    get the list according to the min-distance with item
    """
    # amazon_refurbished_goods_index
    _lst = [ abs(x - item) for x in lst ]

    #ind = _lst.index(min(_lst))
    prices = [x for _, x in sorted(zip(_lst, lst))]

    return prices


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


def accuracy_score(y_test, y_pred):
    accuracy = 0

    num = 0
    i = 0
    while i < len(y_test):
        lst = y_test[i].tolist()
        if y_pred[i] in lst:
            num = num + 1

        i = i + 1
    
    accuracy = num / len(y_pred)

    return accuracy


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

    classifier = SVC(kernel='linear', C=1, random_state=0)
    #scoring = ['precision_macro', 'recall_macro']
    y_pred = cross_val_predict(classifier, X, y_price, cv=10)

    K = 11

    num = 0
    y_test = y_lst[:, 0]
    num = np.sum(y_test == y_pred)

    acc1 = num/len(y_pred)
    print(acc1)

    i = 1
    while i < K:
        y_test = y_lst[:, 0:i+1]
        i = i+1

        #print(confusion_matrix(y_test, y_pred))

        print(accuracy_score(y_test, y_pred))
        #print(recall_score(y_test, y_pred))
        #print(f1_score(y_test, y_pred))

        #print(classification_report(y_test, y_pred))

    
    """
    print(scores.keys())
    print(scores['test_precision_macro'])
    print(scores['test_recall_macro'])
    """

    #

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