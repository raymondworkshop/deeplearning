### deeplearning
 * This is mainly on deeplearning and its applications, especially in NLP & NLU.


#### A Use Case in Intelligent Configurations Design
#####  Descrption 
Online product configurations are prevailing tools in e-commerce industry to elicit customer needs. Yet current online configurations require customers to specify the choices of each product attribute, which poses a great challenge for customers with no background knowledge.

Hence, this project aims to develop a new online product configuration approach which enables customers to configure a product just by specifying their functional requirements, instead of the detailed design parameters.

For example, when users input the keywords like “a large Screen Size laptop“, our approach could map the associated features and and give high quality recommendations <small>(Fig.1: Our approach to predicting the attributes based on the input “A large Screen Size laptop”)<small>.


![Fig.1](https://github.com/muyun/dev.deeplearning/blob/master/nsrc/icon_demo.png) 

##### Solution
We develop tools to extract Amazon user review (laptop) data (suppose these reviews as user inputs). Then we build **Deep learning Models** to do the query-to-attributes mapping [map user inputs (the functional requirements in unstructured query) into product parameters or features (structured attributes)].

##### I gave some Presentations about deep learning
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



   
#### Notes on deep learning
 * ch2 
   - 通过**与非门**的组合可以实现计算机 ? [The elements of computing systems](http://www1.idc.ac.il/tecs/plan.html) 
   - 与非门可以使用感知机实现 => 通过感知机的组合可以表示计算机  
   - 
   - 两层感知机就能构建计算机， 因为两层感知机可以表示任意函数   

 * ch3 
   - 神经网络可以**自动地从数据中学习到合适的权重参数** 
   - 激活函数 必须使用 非线性函数 
   - **batch 批处理** 可以缩短处理时间 
     + 因为数值计算的库都能够高效处理大型数组运算的最优化
     + 批处理一次性计算大型数组要比分开速度更快 - **数据传送**成为瓶颈 

 * ch4 
   - 由人来驱动 - 人来设计算法解决问题，
   - 数据驱动 - 机器学习是由机器**从收集到的数据中找出规律** -> 模型/算法 
   - 
   - 选 mini-batch -> 计算mini-batch的损失函数的值 - 梯度 -> 更新参数 <- loop 
   - 从训练数据中随机选出一部分数据 - mini-batch - 目标是减少mini-batch的损失函数的值 
     + 神经网络 **以损失函数为指标** 寻找最优权重参数 
     +  
     + 如果以识别精度作为指标，参数的导数在绝大多数地方都会变成0 
       - 识别精度**对微小的参数变化**基本上没有什么反应， 它的值是不连续，突然地 变化 

   - **梯度法** - 以梯度的信息为线索，决定前进的方向 
   - **学习率** 由人工设定 - 在一次学习中，在多大程度上更新参数 


#### reference


####  Copyright (c) 2018-2020 [HSUHK](https://stra.hsu.edu.hk/en/)
