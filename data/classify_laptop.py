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
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, recall_score
#from sklearn.metrics.scorer import make_scorer

#from sklearn.model_selection import cross_val_score

def train(file):
    # loading the dataset
    df = pd.read_excel(file)
    df.head()
    df.info()

    #replaced the missing data with the mean of that column
    df.fillna(df.mean(), inplace=True)

    # X - > features, y->label
    y = df['class label']
    X = df.iloc[:, [1,2, 3,4,5]]

    #X_cpu = df['cpu']
    #X_ram = df['RAM']
    #X_hdd = df['HDD']
    #X_monitor = df['monitor']
    #X_graphical = df['graphical']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    """
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)
    acc = svm_model_linear.score(X_test, y_test)
    cm = confusion_matrix(y_test, svm_predictions)
    print(acc)
    print(cm)
    """

    clf = SVC(kernel='linear', C=1, random_state=0)
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, X, y, cv=10, scoring=scoring, return_train_score=False)
    print(scores.keys())
    print(scores['test_precision_macro'])
    print(scores['test_recall_macro'])

    #

    return 0

#
def main():
    #dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    file = "C:\\Users\\raymondzhao\\myproject\\data\\laptop_data.xlsx" 

    train(file)

    print("Done")

#
if __name__ == '__main__':
    main()