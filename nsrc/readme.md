
#### Descrption
Online product configurations are prevailing tools in e-commerce industry to elicit customer needs. Yet current online configurations require customers to specify the choices of each product attribute, which poses a great challenge for customers with no background knowledge.

Hence, this project aims to develop a new online product configuration approach which enables customers to configure a product just by specifying their functional requirements, instead of the detailed design parameters.

For example, when users input the keywords like “a large Screen Size laptop“, our approach could map the associated features and and give high quality recommendations (Fig.1).

![Fig.1](https://github.com/muyun/dev.deeplearning/blob/master/nsrc/icon_demo.png) 

#### Solutions
We develop tools to collect Amazon user reviews (laptop), and suppose these reviews as user inputs. Then we build **Deep learning Model** to do query-to-attributes mapping [map user inputs (the functional requirements in unstructured query) into product parameters or features (structured attributes)].

#### The Presentation about solutions
* [How to collect data as Dataset](https://docs.google.com/presentation/d/1Y7zrC9QLHHcFlQpn3Yb2_paOgU_xB1B4yJwjM6ah98E/edit?usp=sharing)
* [Algorithm: word_embeddings_based_mapping](https://docs.google.com/presentation/d/1XpAfL3T-A0cxyRjVbmZFE2M8jsFQrwycfGX042OSP18/edit?usp=sharing)
* [Algorithm: SVM based on word-embedding](https://drive.google.com/file/d/1GoGhYoFfq1Ha2MoseFj5KcWrzu0IISQ2/view?usp=sharing)
* [CNN Alg for Sentence Classification](https://drive.google.com/file/d/1JKskq_ufcVFyvbG0yfBc1aRl0PLv39ak/view?usp=sharing)
* [The experiments based on CNN Alg](https://drive.google.com/file/d/1JKskq_ufcVFyvbG0yfBc1aRl0PLv39ak/view?usp=sharing)
* [Algorithm: Recurrent Neural Networks](https://drive.google.com/file/d/1UG5GBp7PH-8pOlXFw_jMKGQQpUtEe2xV/view?usp=sharing)
* [Algorithm: LSTM](https://drive.google.com/file/d/1f-5p59g9NrMlYHkhjagAYe7OO-23P-R-/view?usp=sharing)
* [Algorithm:Hier-attention Network](https://drive.google.com/file/d/1MWM-tzy_I7I-MWqkIF3u9KodEKW3K2Tb/view?usp=sharing)
* [Top-N Sort Algorithm](https://drive.google.com/file/d/1kpzEqbFUUvQ3dsSITs5C8ifK0VfOyAB0/view?usp=sharing)
* [TOP-N Alg on word-embedding](https://docs.google.com/presentation/d/1YUsoW0bynIm33QrzEdIuhxSd99JZK30oNusAMOIt3qc/edit?usp=sharing)


#### Notes 
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
         
  
