### work notes

#### 2018-05-23
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