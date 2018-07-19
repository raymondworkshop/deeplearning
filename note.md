### work notes

#### 2018-07-18 
 * 

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
   - GPU: export CUDA_VISIBLE_DEVICES=1
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
> mkvirtualenv tfenv -> including the Python executable files, and a copy of the pip lib  

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
