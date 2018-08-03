"""
data processing and analysis
@raymond

history:
   - create
   - add data stastics on 03/08/2018

"""
from pandas import DataFrame

#import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

import gzip
import json
import ast


# data frame in pandas
df = pd.DataFrame()

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


def get_data(file):
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
                        #
                        if 'weight' in key.lower():
                            asins[_asin].append(value)

                        if len(asins[_asin]) == 5:
                            break
                    
                    reviews = data['reviews']
                    num_words = 0
                    for _t in reviews:
                        t =  " ".join(x.decode("utf-8")  for x in _t)
                        #t =  " ".join(x  for x in _t)
                        num_words = num_words + len(_t)
                        list_reviews.append(t)
                        #reviews = asins[_asin][3] 

                    asins[_asin].append(list_reviews)
                    #num of reviews
                    asins[_asin].append(len(reviews))
                    # num of words
                    asins[_asin].append(num_words)

                # 
                df[_asin] = asins[_asin]

    #return asins
    return df

def analyze_data(df):
    # num of reviews and num of words

    reviews = df.loc[6:7, :]
    #for t in reviews:
    #x_laptops = df #116
    x_reviews = df.loc[6, :].sum() #7227
    y_words = df.loc[7, :].sum()  #520942

    reviews.plot(kind='hist', bins = 120)
    plt.xlabel('num of words')



    return

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

    
    dir = "C:/Users/raymondzhao/myproject/dev.dplearning/data/"
    #dir = "/data/raymond/workspace/exp2/"
    file = dir + 'amazon_reviews.json'
    
    df = get_data(file)
    #save to the csv
    file1 = dir + 'amazon_tech_0803.csv'
    df.to_csv(file1)
    

    #load data
    #file1 = dir + 'amazon_tech_0803.csv'
    #data = pd.read_csv(file1)

    # data statistics
    analyze_data(df)


    #import pdb
    # pdb.set_trace()
    print("Done")


if __name__ == "__main__":
    main()
