# -*- coding: utf-8 -*-
"""
read the data
# update by Raymond
  - top-K alg 


"""

import gzip
import json

import ast
import re

import string

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame()


# refurbished goods?

amazon_refurbished_goods_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                     21, 22, 23, 24, 26, 28, 31, 32, 33, 34, 38, 40, 42, 43, 44, 45,
                     48, 50, 51, 53, 56, 57, 58, 59, 60, 62, 65, 69,70, 72,
                     75, 76, 78, 79, 81, 82, 83, 84, 85, 87, 89, 90, 91, 93,
                     94, 95, 96, 97, 99, 102, 103, 105, 106]
#---------------------------
# The below is to label data  


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


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


def get_cpu_label(lst):
    dict = {}
    #
    ind = 0
    for item in lst:
        if item not in dict:
            dict[item] = ind
            ind += 1
        else:
            pass

    #
    _lst = []
    for item in lst:
        _dict = {}
        if item in dict.keys():
            _dict[item] = dict[item]
            _lst.append(_dict)

    return _lst


def _get_cpu_label(_str):
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


def get_sscreen_label(lst):
    dict = {}
    #
    ind = 0
    for item in lst:
        if item not in dict:
            dict[item] = ind
            ind += 1
        else:
            pass

    #
    _lst = []
    for item in lst:
        _dict = {}
        if item in dict.keys():
            _dict[item] = dict[item]
            _lst.append(_dict)

    return _lst


def _get_sscreen_label(_str):
    # [ 11.6 inches, 13.3 inches, 14 inches, 15.6 inches, 17.3 inches ]
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


def get_text_labels():
    # get documents
    """
    f1 = open('C:/Users/raymondzhao/myproject/dev.dplearning/data/amazon_data_0719.p', 'r')
    asins = pickle.load(f1)
    f1.close()
    """
    #dir = 'C:/Users/raymondzhao/myproject/dev.dplearning/data/'
    #dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    dir = "/data/raymond/workspace/exp2/"
    file = dir + 'amazon_reviews.json'
    asins = _read_data(file)

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
            labels.append(_cpu_id)
            #labels.append(_sscreen_id)
            #labels.append(_ram_id)
            #labels.append(_harddrive_id)
            #labels.append(_graphprocessor_id)
    
    return texts, labels


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


def read_amazon_data(file):
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

    return asins
    #return 0


def _plt():

    lst = [0.2,
0.2,
0.2,
0.2,
0.2,
0.2,
0.2,
0.2,
0.26990291262135924,
0.26990291262135924,
0.26990291262135924,
0.26990291262135924,
0.26990291262135924,
0.2912621359223301,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.5786407766990291,
0.6330097087378641,
0.6330097087378641,
0.6330097087378641,
0.6330097087378641,
0.6330097087378641,
0.6330097087378641,
0.6330097087378641,
0.6330097087378641,
0.6330097087378641,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0
]

    f1 = [0.05058366, 0.0963035,  0.15823606, 0.22616732, 0.29182879, 0.36770428,
 0.43135075, 0.46400778, 0.47557285, 0.48365759, 0.48178281, 0.48476005,
 0.48608201, 0.5086159,  0.50843061, 0.51264591, 0.52941176, 0.52777778,
 0.52631579, 0.525,     0.52380952, 0.52272727, 0.52173913, 0.52083333,
 0.52,       0.51923077, 0.51851852, 0.51785714, 0.51724138, 0.51666667]

    t = np.arange(0, len(f1))
    plt.plot(t, f1, 'bs')
    plt.xlim(0, len(f1))
    plt.ylim(0.2, 1.0)

    #plt.hist(len(lst), lst, color="blue")
    plt.ylabel('f1 (%)')
    plt.xlabel('TOP-N')
    plt.title('The Performance')
    plt.show()

#    print("Num of brands in HP: %d:", num_brands)
#    print("Num of reviews: %d:", num_reviews)
#    print("Num of words: %d:", num_words)

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


def get_amazon_data(file):
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


def get_inds(item, lst):
    """
    get the list according to the min-distance with item
    """
    #_lst = lst
    #_dict = {}

    """
    def dist(x, item):
        return x - item
    """

    # amazon_refurbished_goods_index
    _lst = [ abs(x - item) for x in lst ]

    #ind = _lst.index(min(_lst))
    prices = [x for _, x in sorted(zip(_lst, lst))]

    return prices

#
def sort_items(lst):
    # sort item 

    dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    cpu_tech_file = dir + 'amazon_tech_cpus_1207.json'
    price_lst = []
    #label_lst = []
    asin_map_price = {}
    price_map_label = {}
    #ind = 0  # label
    with open(cpu_tech_file, 'rU') as f1:
        for line in f1:
            if '+' in line:
                #ind += 1
                asin = line.split(':')[0].strip()
                price = int(line.split(':')[3].strip())
                _label = int(line.split(':')[2].strip())
                price_lst.append(price)
                #label_lst.append(label)

                # ind is the label now, use the price as class
                asin_map_price[asin] = price # the duplication
                price_map_label[price] = _label
            #print(ind)
            #print("Done")
        #json.dump(tech_dict, f)

    # map params to prices 
    # get amazon_refurbished_goods_index
    _dict = {} # sort according to the price
    """
    for i in amazon_refurbished_goods_index:
        ind = i - 1
        # in amazon_refurbished_goods_index  
        price = ind_map_price[ind]

        _dict[ind] = get_inds(price, price_lst) 
    """
    
    # to-update
    for asin in asin_map_price:
        price = asin_map_price[asin]
        prices_lst = get_inds(price, price_lst) 
        ind_lst = []
        for price in prices_lst:
            ind_lst.append(price_map_label[price])

        _dict[asin] = ind_lst

    #
    return _dict


def map_cpus_prices(file):

    return 0


def map_params_prices(file):
    #dir = 'C:/Users/raymondzhao/myproject/dev.dplearning/data/'
    #dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    #dir = "/data/raymond/workspace/exp2/"
    #file = dir + 'amazon_reviews.json'

    asins = get_amazon_data(file)

    tech_dict = {}

    _sscreens = []
    _cpus = []
    _rams = []
    _harddrives = []
    _graphprocessors = []
    for _asin in asins:
        print("The asin %s:", _asin)
        # [screensize,cpu, ram, reviews]
        _cpu = asins[_asin][1]
        _cpus.append(_cpu)
        """
        if _cpu:
           #_cpu_id = get_cpu_label(_cpu)
           #labels_index[_cpu] = _cpu_id
           #
           #_cpus.append(_cpu)
           _cpus.append(_cpu)
        """

        _sscreen = asins[_asin][0]
        _sscreens.append(_sscreen)
        """
        if _sscreen:
            #_sscreen_id = get_sscreen_label(_sscreen)
            #labels_index[_sscreen] = _sscreen_id
            #
            _sscreens.append(_sscreen)
        """

        _ram = asins[_asin][2]
        _rams.append(_ram)
        """
        if _ram:
            _ram_id = get_ram_label(_ram)
            labels_index[_ram] = _ram_id
        """
        
        _harddrive = asins[_asin][3]
        _harddrives.append(_harddrive)
        """
        if _harddrive:
            _harddrive_id = get_harddrive_label(_harddrive)
            labels_index[_harddrive] = _harddrive_id
        """
    
        #Graphics Coprocessor
        _graphprocessor = asins[_asin][4]
        _graphprocessors.append(_graphprocessor)
        """
        if _graphprocessor:
            _graphprocessor_id = get_graphprocessor_label(_graphprocessor)
            labels_index[_graphprocessor] = _graphprocessor_id
        """

        tech_dict[str(_asin)] = [_cpu,_sscreen,_ram,_harddrive,_graphprocessor]

    #cpu
    cpu_lst = get_cpu_label(_cpus)
    print(cpu_lst)

    # get the class list according to the dist
    cpu_labels_dict = sort_items(cpu_lst)

    # screen size
    #sscreen_lst = get_sscreen_label(_sscreens)
    #print(sscreen_lst)

    return cpu_labels_dict


def get_amazon_texts_labels(file):
        # get documents
    """
    f1 = open('C:/Users/raymondzhao/myproject/dev.dplearning/data/amazon_data_0719.p', 'r')
    asins = pickle.load(f1)
    f1.close()
    """
    #dir = 'C:/Users/raymondzhao/myproject/dev.dplearning/data/'
    #dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    #dir = "/data/raymond/workspace/exp2/"
    #file = dir + 'amazon_reviews.json'
    asins = get_amazon_data(file)

    #file = dir + 'amazon_tech_all_5.csv'
    #df = pd.read_csv(file)

    # The text samples and their labels
    texts = []  #list of text samples
    #labels = array([])
    labels = [] # list of label ids
    #labels_inds = []
    labels_index = {}  # dictionary mapping label name to numeric id

    # ['14 inches', '2.16 GHz Intel Celeron', '4 GB SDRAM DDR3', 
    # [[b'I', b'placed', b'my', b'order', b'on', b'December', b'19th', b'and', b'was', b'under', b'the', b'impression', b'it', b'would', b'arrive', b'on', b'the', b'22nd']]
        # [screensize,cpu, ram, Hard Drive,Graphics Coprocessor, reviews]
    
    tech_dict = {}

    _sscreens = []
    _cpus = []
    _rams = []
    _harddrives = []
    _graphprocessors = []
    for _asin in asins:
        print("The asin %s:", _asin)
        # [screensize,cpu, ram, reviews]
        _cpu = asins[_asin][1]
        _cpus.append(_cpu)
        """
        if _cpu:
           #_cpu_id = get_cpu_label(_cpu)
           #labels_index[_cpu] = _cpu_id
           #
           #_cpus.append(_cpu)
           _cpus.append(_cpu)
        """

        _sscreen = asins[_asin][0]
        _sscreens.append(_sscreen)
        """
        if _sscreen:
            #_sscreen_id = get_sscreen_label(_sscreen)
            #labels_index[_sscreen] = _sscreen_id
            #
            _sscreens.append(_sscreen)
        """

        _ram = asins[_asin][2]
        _rams.append(_ram)
        """
        if _ram:
            _ram_id = get_ram_label(_ram)
            labels_index[_ram] = _ram_id
        """
        
        _harddrive = asins[_asin][3]
        _harddrives.append(_harddrive)
        """
        if _harddrive:
            _harddrive_id = get_harddrive_label(_harddrive)
            labels_index[_harddrive] = _harddrive_id
        """
    
        #Graphics Coprocessor
        _graphprocessor = asins[_asin][4]
        _graphprocessors.append(_graphprocessor)
        """
        if _graphprocessor:
            _graphprocessor_id = get_graphprocessor_label(_graphprocessor)
            labels_index[_graphprocessor] = _graphprocessor_id
        """

       # tech_dict[str(_asin)] = [_cpu,_sscreen,_ram,_harddrive,_graphprocessor]

    #cpu
    cpu_lst = get_cpu_label(_cpus)
    print(cpu_lst)

    # get the class list according to the dist
    cpu_labels_dict = sort_items(cpu_lst)

    #_sscreens = []
    _cpus = []
    ind = 0
    #for ind in cpu_labels_dict.items():
    for _asin in cpu_labels_dict:
        print("The asin %s:", _asin)
        # [screensize,cpu, ram, reviews]
        _cpu = asins[_asin][1]
        if _cpu:
           #_cpu_id = _get_cpu_label(_cpu)
           _cpu_id = cpu_labels_dict[_asin]
           #_cpus.append(_cpu)

           #labels_index[_cpu] = _cpu_id
           #
           #_cpus.append(_cpu)

        _sscreen = asins[_asin][0]
        if _sscreen:
            _sscreen_id = get_sscreen_label(_sscreen)
            labels_index[_sscreen] = _sscreen_id
            #
            #_sscreens.append(_sscreen)

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
    
    return texts, labels


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
    #file = dir + 'amazon_reviews.json'

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

    #dir = "/data/raymond/workspace/exp2/"
    #file = dir + 'amazon_reviews.json'

    #cpu_dict = map_params_prices(file)

    tech_file =  dir + 'amazon_tech_params_.json'
    cpu_tech_file =  dir + 'amazon_tech_cpus_.json'
    """
    with open(tech_file, 'w') as f:
        for key in tech_dict:
        #json.dump(tech_dict, f)
            f.write(key + " : " + ', '.join(tech_dict[key]) + '\n')
    """

    # get texts and labels
    #dir = 'C:/Users/raymondzhao/myproject/dev.deeplearning/data/'
    #dir = '/data/raymond/workspace/exp2/'
    #file = 'amazon_reviews.json'
    #file = 'amazon_reviews_copy.json'
    reviews = []
    #texts, labels = get_amazon_texts_labels(file)

    _plt()

 
    #asins = read_data(file)

    #read_flipkart_data(file4,csv_file4)

    #analyze_data(df, file2)

    #import pdb
    # pdb.set_trace()
    print("Done")


if __name__ == "__main__":
    main()
