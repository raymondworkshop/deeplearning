# -*- coding: utf-8 -*-
"""
read the data

"""

import gzip
import json

import ast
import re

import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame()


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


def get_data(file, file2):

    list_reviews = []
    asins = {}

    len_reviews = []

    dict = {}

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
                        # hard drive
                        if 'hard' in key.lower():
                            asins[_asin].append(value)
                        #
                        if 'weight' in key.lower():
                            asins[_asin].append(value)

                        if len(asins[_asin]) == 5:
                            break

                    reviews = data['reviews']
                    texts = []
                    num_words = 0
                    for _t in reviews:
                        t = [w.decode("utf-8") for w in _t]
                        s = " ".join(x for x in t)
                        #t =  " ".join(x  for x in _t)
                        num_words = num_words + len(_t)
                        texts.append(s)
                        #reviews = asins[_asin][3]

                        len_reviews.append(len(_t))

                    while len(texts) < 800:
                        x = " "
                        texts.append(x)

                    asins[_asin].append(texts)
                    # num of reviews
                    asins[_asin].append(len(reviews))
                    # num of words
                    asins[_asin].append(num_words)

                    #
                    dict[_asin] = texts

                #
                #df[_asin] = asins[_asin]

                   # write to excel
                    writer = pd.ExcelWriter(file2, engine='xlsxwriter')

                    df1 = pd.DataFrame(dict)
                    df1.to_excel(writer, sheet_name=_asin)
                    writer.save()

                    writer.close()

    # return asins
    return 0


def analyze_data(df, file2):
    dict = {}

    asins = df.keys()
    for asin in asins[1:]:
        texts = df[asin][5]
        dict[asin] = texts

    df1 = pd.DataFrame.from_dict(dict, orient='index')

    df1.to_csv(file2, sep='\t')

    return 0


def read_data(file):
    #dir = "/data/raymond/workspace/exp2/"
    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'

    file = dir + 'amazon_tech_all_5.csv'
    df = pd.read_csv(file)

    asins = {}

    texts = []
    _asins = df.columns.values.tolist()[1:]

    df.loc[df[5].isin(_asins)]
    reviews = df.loc[5, :].tolist()
    for _t in reviews[1:]:
        t = [w.decode("utf-8") for w in _t]
        s = " ".join(x for x in t)
        #
        texts.append(s)

    return 0


def _read_data(file):
    #list_reviews = []
    asins = {}

    f1 = open('amazon_only_reviews.txt', 'w', encoding="utf-8")

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
                f1.write(_asin + ":" + "\t")

                reviews = data['reviews']
                texts = []
                for review in reviews:
                    # list_reviews.append(review)
                    stripped = [w.decode("utf-8") for w in review]
                    s = " ".join(x for x in stripped)
                    texts.append(s)
                    # write file in some format
                    #s.decode("cp950", "ignore")
                    f1.write(s + '\n')

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
                        # Hard Drive
                        if 'hard' in key.lower():
                            asins[_asin].append(value)
                        # weight
                        if 'weight' in key.lower():
                            asins[_asin].append(value)

                        if len(asins[_asin]) == 5:
                            break

                asins[_asin] = texts
    f1.close()
    # return asins
    return 0


def read_hp_data(file3):
    hp_asins = {}

    num_reviews = 0
    num_words = 0
    num_brands = 0

    words = []
    reviews_ = []

    sheets = []
    xlsx = pd.ExcelFile(file3)
    for sheet in xlsx.sheet_names[1:]:
        # sheets.append(xlsx.parse(sheet))
        content = xlsx.parse(sheet)
        #parameters = content[0].split('\n')
        #processor = parameters[0]
        reviews = content[1:]
        #hp_asins[sheet].append(parameters)
        #hp_asins[sheet].append(reviews)
        
        _reviews = reviews.values.tolist()
        _num_reviews = 0
        for review in _reviews:
            #stripped = [w for w in review[0]]
            #s = " ".join(x for x in stripped)
            #texts.append(s)

            num_reviews += 1
            _num_reviews += 1
            if not isinstance(review[0] , float):
                num_words = num_words + len(review[0].split()) 
                words.append(len(review[0].split()))
        
        reviews_.append(_num_reviews)

        # hp_asins[sheet_names]
        num_brands = num_brands + 1

    #num_reviews = num_reviews + _num_reviews
    #num_words = num_words + _num_words
    #num_brands += 1
    # hist
    """
    #plt.hist(len_reviews, bins=20, color='g')
    plt.hist(words,bins=40, color="blue")
    plt.xlabel('Number of words in one review')
    plt.ylabel('Number of reviews')
    plt.title('The distribution of the number of words in the review')
    plt.show()
    """

    plt.hist(reviews_, bins=20, color="blue")
    plt.xlabel('Number of reviews in one laptop')
    plt.ylabel('Number of laptops')
    plt.title('The distribution of the number of reviews')
    plt.show()

    print("Num of brands in HP: %d:", num_brands)
    print("Num of reviews: %d:", num_reviews)
    print("Num of words: %d:", num_words)

    return hp_asins


def read_flipkart_data(file, csv_file4):
    flipkart_asins = {}

    num_brands = 0
    num_reviews = 0
    num_words = 0

    reviews = []
    words = []

    dict = {}
    #writer = pd.ExcelWriter(csv_file4, engine='xlsxwriter')

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
            if 'tech' in line and 'reviews' in line:
                _asin = str(data.keys())
                value = list(data.values())[0]
                #f1.write(_asin + ":" + "\t")

                _reviews = value['reviews']
                texts = []
                _num_reviews = 0
                _num_words = 0
                for review in _reviews:
                    # list_reviews.append(review)
                    stripped = [w.decode("utf-8") for w in review]
                    s = " ".join(x for x in stripped)
                    texts.append(s)

                    _num_reviews += 1
                    _num_words = _num_words + len(stripped)

                    words.append(len(stripped)) # word list
                    # write file in some format
                    #s.decode("cp950", "ignore")
                    #f1.write(s + '\n')

                params = value['tech']
                if len(params) > 0:
                    #asins[_asin] = [screensize,cpu, ram]
                    flipkart_asins[_asin] = params

                flipkart_asins[_asin] = _num_reviews
                flipkart_asins[_asin] = _num_words
                flipkart_asins[_asin] = texts

                reviews.append(_num_reviews) #review list

                dict[_asin] = texts

            num_reviews = num_reviews + _num_reviews
            num_words = num_words + _num_words
            num_brands += 1


    # write to excel
    #writer = pd.ExcelWriter(csv_file4, engine='xlsxwriter')
                           # write to excel
            writer = pd.ExcelWriter(csv_file4, engine='xlsxwriter')

            df1 = pd.DataFrame.from_dict(dict, orient='index').transpose()
            df1.to_excel(writer, sheet_name=_asin[13:41])
            writer.save()

            writer.close()
    

    # hist
    #plt.hist(len_reviews, bins=20, color='g')

    #plt.hist(len_reviews, bins=20, color='g')
    """
    plt.hist(words,bins=40, color="blue")
    plt.xlabel('Number of words in one review')
    plt.ylabel('Number of reviews')
    plt.title('The distribution of the number of words in the review')
    plt.show()
    """

    #
    """
    plt.hist(reviews, bins=20, color="blue")
    plt.xlabel('Number of reviews in one laptop')
    plt.ylabel('Number of laptops')
    plt.title('The distribution of the number of reviews')
    plt.show()
    """


    # f1.close()
    # return asins
    print("Num of brands in Flipkart: %d:", num_brands)
    print("Num of reviews: %d:", num_reviews)
    print("Num of words: %d:", num_words)
    return 0


def main():
    """
    path = "/Users/zhaowenlong/workspace/proj/dev.dplearning/data/reviews_Electronics_5_small.json.gz"
    f = open(
        "/Users/zhaowenlong/workspace/proj/dev.dplearning/data/output.strict", 'w')
    for l in parse(path):

        #import pdb
        # pdb.set_trace()
        txt = json.loads(l)
        f.write(txt["reviewText"] + '\n')
    """

    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    #dir = "/data/raymond/workspace/exp2/"
    file = dir + 'amazon_reviews_copy.json'

    file2 = dir + 'reviews.xls'

    #df = get_data(file, file2)
    #file1 = dir + 'amazon_tech_0903.csv'
    # df.to_csv(file1)

    #df = pd.read_csv(file1)

    # load excel
    file3 = dir + 'hp_laptop.xlsx'
    #hp_asins = read_hp_data(file3)

    file4 = dir + 'flipkart_reviews_1005.json'

    csv_file4 = dir + 'flipkart_reviews.xlsx'

    read_flipkart_data(file4,csv_file4)

    #analyze_data(df, file2)

    #import pdb
    # pdb.set_trace()
    print("Done")


if __name__ == "__main__":
    main()
