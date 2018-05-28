### work notes

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
