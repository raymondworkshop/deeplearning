
#### descrption
Online product configurations are prevailing tools in e-commerce industry to elicit customer needs. Yet current online configurations require customers to specify the choices of each product attribute, which poses a great challenge for customers with no background knowledge.

Hence, this project aims to develop a new online product configuration approach which enables customers to configure a product just by specifying their functional requirements, instead of the detailed design parameters.

For example, when users input the keywords like “a large Screen Size laptop“, our approach could map the associated features and and give high quality recommendations (Fig.1).

![Fig.1](https://github.com/muyun/dev.deeplearning/blob/master/nsrc/icon_demo.png) 

#### solutions
We develop tools to collect Amazon user reviews (laptop), and suppose these reviews as user inputs, and then use **Deep learning** technology to map user inputs (the functional requirements in unstructured query) into product parameters or features (structured attributes)


#### notes 
* requirements 
  - Install the related pkgs in requirements.txt 
  - Pre-trained [GloVe word vectors](https://nlp.stanford.edu/projects/glove/)

* dataset
  - scrape review data from [Amazon](www.amazon.com) -  follow the instructions
    + code -> [scraper_amazon.py](https://github.com/muyun/dev.deeplearning/blob/master/src/scraper_amazon.py) 
  - other review data

* algs - NN modules 
   - MLP -> trainsMLP.py 
   - RNN -> textClassifierRNN.py 
   - CNN -> textClassifierCNN.py 
   - Attention -> textClassifierHATT.py 
         
  
