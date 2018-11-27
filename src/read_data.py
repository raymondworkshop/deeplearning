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
import numpy as np

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

    len_reviews = []

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
                    num_words = 0
                    for _t in reviews:
                        t = " ".join(x.decode("utf-8") for x in _t)
                        #t =  " ".join(x  for x in _t)
                        num_words = num_words + len(_t)
                        list_reviews.append(t)
                        #reviews = asins[_asin][3]

                        len_reviews.append(len(_t))

                    asins[_asin].append(list_reviews)
                    # num of reviews
                    asins[_asin].append(len(reviews))
                    # num of words
                    asins[_asin].append(num_words)

                #
                df[_asin] = asins[_asin]

    # return asins
    return df, len_reviews


def analyze_data(df, len_reviews):
    # num of reviews and num of words

    reviews = df.loc[6:7, :]
    # for t in reviews:
    # x_laptops = df #116
    # x_reviews = df.loc[6, :].sum()  # 7227
    # y_words = df.loc[7, :].sum()  # 520942

    x_words = df.loc[7, :].tolist()
    y_reviews = df.loc[6, :].tolist()

    # Todo use pandas
    #reviews.plot(kind='hist', bins=120)

    # pdb.set_trace()

    #plt.hist(len_reviews, bins=20, color='g')
    plt.hist(y_reviews, bins=20, color="green")

    plt.xlabel('Number of words in one review')
    plt.ylabel('Number of reviews')
    plt.title('The distribution of the number of words in the review')
    plt.show()

    import pdb
    pdb.set_trace()

    return


def plot():
     
    acc = [0.13494809689612536, 0.48373702426270215, 0.5211072664359861, 0.5619377163042246, 0.47889273357432605, 0.4823529412589684, 0.505882352961801, 0.4809688581521123, 0.4899653979444999, 0.4837370242420777, 0.47543252597218155, 0.48927335642200853, 0.49342560555695664, 0.4858131488095518, 0.49134948098948256, 0.49134948098948256, 0.5003460207818701, 0.49757785469190474, 0.4920415225119739, 0.49134948098948256, 0.4830449827195864, 0.4858131488095518, 0.48442906576456907, 0.4927335640344653, 0.4809688581521123, 0.48235294119709504, 0.48235294119709504, 0.4768166090171642, 0.48650519033204315, 0.45605536334242375, 0.4581314879098978, 0.45328719725245836, 0.47958477510712966, 0.4816608996746037, 0.4920415225119739, 0.4837370242420777, 0.47266435988221617, 0.47404844292719883, 0.4761245674946729, 0.4837370242420777]
    precision =  [0.0, 0.5155326641831831, 0.55124750962392, 0.6019534415084601, 0.5028904746920381, 0.48663639557766447, 0.515937508419582, 0.48891083646702166, 0.49603288544364943, 0.4926963517782229, 0.4800386947948061, 0.49139977393823464, 0.49980198426291944, 0.4907929078848709, 0.4983768970520878, 0.49670871004948475, 0.5078421878468067, 0.5008237526509655, 0.49767330867438264, 0.49737863292932166, 0.4892774839439146, 0.49470268772299875, 0.4928386293992831, 0.4972659282910965, 0.4872111964010306, 0.4861339240924401, 0.48563984332200455, 0.48288903050587506, 0.4914812984017137, 0.4585990328867267, 0.46104057693220585, 0.45660735432038413, 0.4840846102230725, 0.4872455233483827, 0.49384298586280023, 0.48686386129404313, 0.4773017821929532, 0.4779953119767831, 0.47898947055102875, 0.4854677695196379]
    recall = [0.0, 0.1903114186851211, 0.41730103806228375, 0.4934256055363322, 0.4408304498269896, 0.45674740484429066, 0.4961937716262976, 0.4726643598615917, 0.48373702422145326, 0.47612456747404847, 0.46643598615916954, 0.4795847750865052, 0.4885813148788927, 0.4754325259515571, 0.485121107266436, 0.48304498269896196, 0.4961937716262976, 0.49204152249134947, 0.4865051903114187, 0.485121107266436, 0.4726643598615917, 0.4802768166089965, 0.47681660899653977, 0.485121107266436, 0.4726643598615917, 0.473356401384083, 0.47681660899653977, 0.471280276816609, 0.4802768166089965, 0.4477508650519031, 0.45121107266435984, 0.4470588235294118, 0.473356401384083, 0.47612456747404847, 0.4844290657439446, 0.4795847750865052, 0.4685121107266436, 0.4692041522491349, 0.4698961937716263, 0.47681660899653977]
    f1 = [0.0, 0.3529220414341521, 0.48427427384310184, 0.5476895235223962, 0.47186046225951384, 0.47169190021097757, 0.5060656400229397, 0.4807875981643067, 0.48988495483255134, 0.4844104596261357, 0.47323734047698784, 0.4854922745123699, 0.4941916495709061, 0.483112716918214, 0.49174900215926187, 0.48987684637422335, 0.5020179797365522, 0.49643263757115746, 0.49208924949290067, 0.4912498700978788, 0.48097092190275315, 0.48748975216599766, 0.4848276191979114, 0.49119351777876624, 0.47993777813131117, 0.47974516273826157, 0.4812282261592722, 0.47708465366124203, 0.4858790575053551, 0.4531749489693149, 0.45612582479828284, 0.45183308892489793, 0.4787205058035777, 0.4816850454112156, 0.48913602580337245, 0.48322431819027417, 0.4729069464597984, 0.473599732112959, 0.47444283216132754, 0.4811421892580888]
    
    x = np.arange(len(f1))
    plt.title('Performance')
    plt.plot(x, acc, color='green', label='accuracy')
    plt.plot(x, precision, color='cyan', label='precision')
    plt.plot(x, recall, color='magenta', label='recall')
    plt.plot(x, f1, color='blue', label='f1')
    plt.legend()

    plt.xlabel('Number of epochs')
    plt.ylabel('value')
    plt.show()

    #import pdb
    #pdb.set_trace()
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

    #dir = "C:/Users/raymondzhao/myproject/dev.dplearning/data/"
    #dir = "/data/raymond/workspace/exp2/"
    dir = "/Users/zhaowenlong/workspace/proj/dev.deeplearning/data/"
    file = dir + 'amazon_reviews.json'

    #df, len_reviews = get_data(file)
    # save to the csv
    file1 = dir + 'amazon_tech_0803.csv'
    #df.to_csv(file1)

    # load data
    #file1 = dir + 'amazon_tech_0803.csv'
    #data = pd.read_csv(file1)

    # data statistics
    #analyze_data(df, len_reviews)



    # plot
    plot()

    #import pdb
    # pdb.set_trace()
    print("Done")


if __name__ == "__main__":
    main()
