"""
word embedding using gensim lib
@author wenlong
"""
import gensim
import logging

import json
import gzip

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
    sentences = Sentences()
    
    """
    sentences = []
    for review in parse('C:/Users/raymondzhao/myproject/research/src/reviews_Electronics_big.json.gz'):
        sentences.append(review['reviewText'])
    """

    #training
    #parameter - 
    model = gensim.models.Word2Vec(sentences, size=300, min_count=1) #
    model.save('amazon_reviews.model')
    model.save_word2vec_format('amazon_reviews.model.bin', binary=True)

    #test
    w1 = "good"
    model.wv[w1]

def main():
    word2vec_training()

if __name__ == "__main__":
    main()

