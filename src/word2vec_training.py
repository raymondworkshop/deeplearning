"""
word embedding using gensim lib
@author wenlong
"""
import gensim
import logging

import json
import gzip
import ast

class Sentences():
    def __iter__(self):
        with open('C:/Users/raymondzhao/myproject/research/src/reviews_Electronics_big.json', 'rb') as f:
            for i, line in enumerate(f):
                if (i % 10000 == 0):
                    logging.info("read {0} reivews".format(i))
                data = eval(line.decode("utf-8"))
                #print(data['reviewText'])
                yield data['reviewText'].split()

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def word2vec_training():
    #sentences = word2vec.Text8Corpus('reviewsText_small.zip')
    #sentences = Sentences()
    
    """
    sentences = []
    for review in parse('C:/Users/raymondzhao/myproject/research/src/reviews_Electronics_big.json.gz'):
        sentences.append(review['reviewText'])
    """
    sentences = []
    with open("C:\\Users\\raymondzhao\\myproject\\amazon_reviews.json", "r", encoding="utf-8") as f1:
        for line in f1:
            if 'reviews' in line:
                #line = line.replace("\'", "\"")
                data = ast.literal_eval(line)
                reviews = data['reviews']

                for _review in reviews:
                    review = [w.decode() for w in _review]
                    sentences.append(review)
     
    #training
    #parameter - 
    model = gensim.models.Word2Vec(sentences, size=300, min_count=1) #
    model.save('amazon_reviews_only.model')
    model.save_word2vec_format('amazon_reviews_only_model.bin', binary=True)

    #test
    w1 = "good"
    model.wv[w1]

def main():
    word2vec_training()

if __name__ == "__main__":
    main()

