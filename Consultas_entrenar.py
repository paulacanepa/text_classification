#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from unicodedata import normalize
from wordcloud import WordCloud
import tensorflow as tf
import html
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import seaborn as sns
import matplotlib.pyplot as ptl
import nltk
from nltk import SnowballStemmer


# In[ ]:


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


# In[ ]:


def armar_bag_of_words(dataset):
    corpus = []
    all_stopwords = stopwords.words('spanish')
    all_stopwords.extend(("saludo","dia", "noche", "noches", "tardes", "buenos", "buenas", "atentamente", "dias", "estimado", "estimados", "estimada", "atte", "hola", "gracia", "caja", "respuesta", "adjunto", "mucha", "me", "cordoba", "buen", "ud"))
    removeList=["no", "nunca"]
    all_stopwords = [e for e in all_stopwords if e not in removeList]
    for i, value in dataset.items():
        review = str(html.unescape(dataset[i]))
        review = cleanhtml(review)
        review = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
            normalize( "NFD", review), 0, re.I)
        review = normalize( 'NFC', review)
        review = re.sub('[^a-zA-Zá-ú0-9]', ' ', review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

# In[ ]:


def bag_of_words_spacy2(dataset):
    import spacy
    nlp = spacy.load('es_core_news_md')
    spanishstemmer=SnowballStemmer("spanish")
    all_stopwords = stopwords.words('spanish')
    all_stopwords.extend(("saludo","dia", "noche", "noches", "tardes", "buenos", "buenas", "atentamente", "dias", "estimado", "estimados", "estimada", "atte", "hola", "gracia", "caja", "respuesta", "adjunto", "mucha", "me", "cordoba", "buen", "ud"))
    removeList=["no", "nunca"]
    all_stopwords = [e for e in all_stopwords if e not in removeList]
    corpus = []
    for i, value in dataset.items():
        review = str(html.unescape(dataset[i]))
        review = cleanhtml(review)
        review = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize( "NFD", review), 0, re.I)
        review = normalize( 'NFC', review)
        review = re.sub('[^a-zA-Zá-ú0-9]', ' ', review)
        review = review.lower()
        doc = nlp(review)
        stems = [spanishstemmer.stem(token) for token in doc if not token in set(all_stopwords)]
        review = ' '.join(stems)
        corpus.append(review)
    return corpus


# In[ ]:


def split_dataset(X, y, size):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)
    return X_train, X_test, y_train, y_test


# In[ ]:


def top_palabras(corpus):
    plt.figure(figsize = (20,20)) 
    wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(corpus))
    plt.imshow(wc,interpolation = 'bilinear')


# In[ ]:



def contar_palabras(corpus, max_p):
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    cv = CountVectorizer(max_features = max_p)
    X = cv.fit_transform(corpus).toarray()
    pickle.dump(cv,open("countVectorizer","wb"))
    return X


# In[ ]:


def oversampling(data, tipologia):
    for i in range(0, len(tipologia)):
        cant=1000-len(data[data['RECTIPID']==tipologia[i]])
        oversample=data[data['RECTIPID']==tipologia[i]].sample(n=cant, replace=True, random_state=1)
        data=data.append(oversample)
    return data


# In[ ]:


def create_model(n_salida, n_oculto):
    tf.keras.backend.clear_session()
    ann = tf.keras.models.Sequential()
    for i in range(0, len(n_oculto)):
      ann.add(tf.keras.layers.Dense(units=n_oculto[i], activation='relu'))
    ann.add(tf.keras.layers.Dense(units=n_salida, activation='softmax'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann


# In[ ]:


def create_model2(nodo_out):
    tf.keras.backend.clear_session()
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=30, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=30, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=nodo_out, activation='softmax'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann


# In[ ]:


def bag_of_words_spacy(dataset):
    import spacy
    spanishstemmer=SnowballStemmer("spanish")
    nlp = spacy.load('es_core_news_md')
    nlp.Defaults.stop_words |= {"saludo","dia", "noche", "noches", "tardes", "buenos", "buenas", "atentamente", "dias", 
                               "hola", "estimado", "estimados", "estimada", "atte"}
    nlp.Defaults.stop_words -= {"no", "nunca"}
    corpus = []
    for i, value in dataset.items():
        review = str(html.unescape(dataset[i]))
        review = cleanhtml(review)
        doc = nlp(review)
        words = [t.orth_.lower() for t in doc if not t.is_punct | t.is_stop] #elimina signos de puntuacion y stopwords
        #lexical_tokens = [t.lower() for t in words if len(t) > 2 and t.isalpha()] # pasa a minuscula, elimina pal de 2letras y num
        review = ' '.join(words)
        doc=nlp(review)
        lemmas = [tok.lemma_.lower() for tok in doc]
        stems = [spanishstemmer.stem(token) for token in lemmas]
        review = ' '.join(stems)
        corpus.append(review)
    return corpus

# In[ ]:

def spacy_lematizar(dataset, tipo_palabra):
    import spacy
    nlp = spacy.load('es_core_news_md')
    all_stopwords = stopwords.words('spanish')
    removeList=["no", "nunca"]
    all_stopwords = [e for e in all_stopwords if e not in removeList]
    corpus = []
    for i, value in dataset.items():
        review = str(html.unescape(dataset[i]))
        review = cleanhtml(review)
        review = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize( "NFD", review), 0, re.I)
        doc = nlp(review)
        lemmas = [tok.lemma_.lower() for tok in doc if not tok in set(all_stopwords) and tok.pos_ in set(tipo_palabra) ]
        review = ' '.join(lemmas)
        corpus.append(review)
    return corpus