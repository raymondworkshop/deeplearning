### work notes

#### 2018-11-01 
 * todo 
   - RNN-LSTM ?
 * 

#### 2018-10-11 
 * summary
  - 
 * todo 
  - pre-processing 
  - 

#### 2018-10-08 
 * summary
  - flipkart
  Num of brands in Flipkart: %d: 408
  Num of reviews: %d: 32434 - 32k
  Num of words: %d: 1171703 - 1.17M

  - HP
  Num of brands in HP: %d: 59
  Num of reviews: %d: 950 - 1k
  Num of words: %d: 78173 - 18k

#### 2018-10-05
 * flipkart
  - 408 brands

#### data 
 * HP
  - class="bv-content-container"

#### 2018-09-07 
 * ch6 
  - 最优化 -> 神经网络学习的目的是找到使损失函数的值尽可能小的参数 

  - 权重的初始值 - 权值衰减 -> 减少权重参数的值可以抑制过拟合的发生  ？

  - 


 * more correct labels,  lower metrics 

 * experiments on cpu 

1. Shape of data tensor: (7227, 108)
[[   0    0   10    0]
 [   0   26   12    1]
 [   3    3 1207   20]
 [   0    0   41  122]]
0.9377162629757786
             precision    recall  f1-score   support

          0       0.00      0.00      0.00        10
          1       0.90      0.67      0.76        39
          2       0.95      0.98      0.96      1233
          3       0.85      0.75      0.80       163

avg / total       0.93      0.94      0.93      1445


2. Classify ...
[[   0    0   10    0    0]
 [   0   28   10    1    0]
 [   1   13 1200   16    3]
 [   0    0   39  124    0]
 [   0    0    0    0    0]]
0.9356401384083045
/usr/local/tensorflow/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
             precision    recall  f1-score   support

          0       0.00      0.00      0.00        10
          1       0.68      0.72      0.70        39
          2       0.95      0.97      0.96      1233
          3       0.88      0.76      0.82       163
          4       0.00      0.00      0.00         0

avg / total       0.93      0.94      0.93      1445


#### 2018-09-07 

##### ch5 
 * deep learning
   random mini-batch -> W's gradients -> update W -> repeat


 * why to use loss function in deep learning ? 
   - 

 * backward alg could compute gradients effectively 
   - 数值微分耗时 -> 确认BP的实现是否正确 

 * TODO - the code



#### 2018-08-30 
a fast pc -> 2.16 GHz Intel Celeron -> latop


#### 2018-08-24 
  * modify the classifier

  * remove Punctuation 
    - todo: stemming
    - -> acc: 0.4738 - precision: 0.2292 - recall: 0.0311
      -> acc: 0.4743 - precision: 0.2507 - recall: 0.0301


#### 2018-08-22 
 * re-code SWEM-max 
   - L patch ?

 * re-code SWEM-aver + SWEM-conct 
 *  

#### 2018-08-15 
 * SWEM alg improvements

#### 2018-08-06
 * some about SWEM

 * need to learn PyTorch on fast.ai


 * Some on Keras
   - 符号计算 - 计算图
   - Model - 
   - batch - batch_size -> mini-batch gradient descent 
   - epochs


#### 2018-08-03
  * use Pandas lib for data

  * todo
    - PyTorch


#### 2018-08-01
  * research
    - 学会找出研究人员和论文的基本出发点 - motivation

  * Pandas

  * todo:
    - Latex
    - data Visualisation


#### 2018-07-18 
 * result

Found 838332 reviews.
Found 21616 unique tokens.
Shape of data tensor: (838332, 300)
Shape of label tensor: (838332, 6)

SWEM-aver:

Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 300)               0
_________________________________________________________________
embedding_1 (Embedding)      (None, 300, 100)          2000000
_________________________________________________________________
lambda_1 (Lambda)            (None, 100)               0
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 606
=================================================================
Total params: 2,000,606
Trainable params: 606
Non-trainable params: 2,000,000
_________________________________________________________________
None
Train on 528149 samples, validate on 58684 samples


#### 2018-07-17 
 * work harder

 * problem fix:
  - export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
 

#### 2018-07-13  
 *　精力问题

#### 2018-07-12  
 * 

#### 2018-07-11
 * notes on the server
   - source /usr/local/tensorflow/bin/activate
   - GPU: export CUDA_VISIBLE_DEVICES=0
   - check gpu process : nvidiva-smi


#### 2018-07-09 
 * reproduce the paper



#### 2018-05-31
  * read the paper - "predicting latent structured intents from shopping queries"     - MLP
    - LSTM/RNN
    - CNN
    
  * claw the review data - "small" subsets  
  * run basic word2vec on TF  

#### 2018-05-30
  * word embedding
    + data normalization  
      - lemmatization and word stem ?  
      
    + document vectorization  
      - count vectorizer and TF-IDF vectorizer  
      - word2vec <- could care about the order (word context) -> semantic  
        -> we get the similar vectors for the words  
        
    + 


#### 2018-05-29
  * setup google cloud <http://cs231n.github.io/gce-tutorial/>
  
  * NN algorithms
  
  * the server: 10.237.4.253 raymond/raymond

#### meetup with Dr. Wang on 28/05/2018
* about tensorflow [1]:
    + Setup the env about python and tensorflow
    + now a simple tensorflow could be run in locally -> see the github [4] 
    
    + plan (todo):
        - run a basic mode like linear regression in tensorflow
        - then run word2vec in tensorflow, or try to run it on server
    
* about word embedding [2][3]:
    +  the basic applied machine learning knowledge:  like loss functions, bag of words, features, bag of vectors 
       - if there is something wrong, if we could know the principle/theory, we could know the reason and correct it quickly
 
    + plan (todo):
       - know more about ML, especially deep learning (like word embedding part) based on the reference 2 and 3

* Others
   + the github for the code
   + the fixed meetting time


#### 2018-05-24
  * setup the env on python and tensorflow on mac
  * TODO: 
    - setup the env on windows


#### virtualenvwrapper -> export WORKON_HOME=$HOME/.env

  * For some project 
> cd dev.dplearning  

  * create an env 
> <del>mkvirtualenv tfenv -> including the Python executable files, and a copy of the pip lib  </del>  
> python3 -m venv ~/.env/isightenv 
> 
  * use the env
> workon tfenv  -> using the virtual env  
> pip install * 

  * deactivate 
> deactivate -> deactivate the env  
> rmvirtualenv venv

> 
> pip freeze > requirements.txt 

> pip install -r requirements.txt

#### reference
* [Pipenv & Virtual Environments](http://docs.python-guide.org/en/latest/dev/virtualenvs/#lower-level-virtualenv)
