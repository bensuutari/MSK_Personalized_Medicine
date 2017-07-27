#      Kaggle: Memorial Sloan Kettering Personalized Medicine Competition        #


This is my submission for Memorial Sloan Kettering's "Redefining Cancer Treatment" Kaggle competition (https://www.kaggle.com/c/msk-redefining-cancer-treatment).

Here I am using Gensim's (https://radimrehurek.com/gensim/) Word2Vec library to break down text blocks into word vectors.  Following Word2Vec, I implement a Support Vector Machine (SVM) to classify experts' text blocks by their resepective labels.

Genes identified and genetic variations in each case will need to be taken into account going forward, but for now, this code is focused on classifying the expert text blocks.  

**This is a work in progress.**

Next steps:

-Integrate genes and genetic variations into SVM classifier or consider using a different classifier

-Consider using Doc2Vec instead of Word2Vec (info here: https://radimrehurek.com/gensim/models/doc2vec.html)


Tools used: NLTK, Gensim, Scikit-Learn, Pandas, BeautifulSoup, Numpy


