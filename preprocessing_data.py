# output return all the vaocabulary from given text data

import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import matplotlib.pyplot as plt

import string
import re
import gensim
import nltk
from gensim.parsing.preprocessing import STOPWORDS
lemmatizer=WordNetLemmatizer()
seed = 42
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import math
from wordcloud import WordCloud
import streamlit as st
from streamlit import components
import io
import requests
from nltk.util import ngrams
import pyLDAvis
import pyLDAvis.sklearn
# preprocessing part

nltk.download('punkt')
nltk.download('wordnet')
from gensim.parsing.preprocessing import STOPWORDS

seed = 42


#Expand the reviews x is aninput string of any length. Convert all the words to lower case
def preprocessing(text):
    text = str(text).lower()
    text = text.replace(",000,000", " m").replace(",000", " k").replace("′", "'").replace("’", "'")\
                           .replace("won't", " will not").replace("cannot", " can not").replace("can't", " can not")\
                           .replace("n't", " not").replace("what's", " what is").replace("it's", " it is")\
                           .replace("'ve", " have").replace("'m", " am").replace("'re", " are")\
                           .replace("he's", " he is").replace("she's", " she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will").replace("how's"," how has").replace("y'all"," you all")\
                           .replace("o'clock"," of the clock").replace("ne'er"," never").replace("let's"," let us")\
                           .replace("finna"," fixing to").replace("gonna"," going to").replace("gimme"," give me").replace("gotta"," got to").replace("'d"," would")\
                           .replace("daresn't"," dare not").replace("dasn't"," dare not").replace("e'er"," ever").replace("everyone's"," everyone is")\
                           .replace("'cause'"," because").replace("i've"," i have").replace("ive"," i have").replace("don't","do not").replace("dont","do not")
    
    text = re.sub(r'&lt;', ' ', text) #remove '&lt;' tag
    text = re.sub(r'[^\x00-\x7f]', ' ', text) #remove non ASCII strings
    text = re.sub(r'<.*?>', ' ', text) 
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www.\S+", " ", text)
    
    text = re.sub(r'&lt;', ' ', text) #remove '&lt;' tag
    text = re.sub(r'<.*?>', ' ', text) # remove html
    text = re.sub(r'[0-9]+', ' ', text) #remove number
    text = re.sub(r'[^\w\s]', ' ', text) #remove punctiation    
    
    for c in ['\r', '\n', '\t'] :
        text = re.sub(c, ' ', text) #replace newline and tab with tabs\
        text = re.sub('\s+', ' ', text) #replace multiple spaces with one space
    
    custom_stopword = ['good','better','card','like','right','get','come','want','trying','look','think','need',
                                            'gpus','subreddit', 'haveseek','haveremaining','ratechapterschaptersdescriptionsdescriptions',
                                            'came','seeing','mentioned','offering','offered','left','got','said','took', 'selectedcaptionscaptions','havel',
                                            'ti','looking', 'm','help','tried','know','welcome','potato','u','varies','visiin',
                                            'comment','buildapcforme','sure','amd','gpu','cpu','having','have',
                                            'getting','yes','lot','thanks','going','dollar', 'pm']
    # custom_stopword = custom_stopword.append(new_sw)

    all_stopwords_gensim = STOPWORDS.union(set(custom_stopword))
    tokens_vs_stop = word_tokenize(text)
    text = ' '.join([word for word in tokens_vs_stop if not word in all_stopwords_gensim])
        
    return text

# import spacy
# spacy.cli.download("en")
# nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

# def posta(texts, allowed_postags=['NOUN', 'VERB']):  # ['NOUN', 'ADJ', 'VERB', 'ADV']
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent))
#         texts_out.append([token for token in doc if token.pos_ in allowed_postags]) # sửa, bỏ lemma đi
#     return texts_out


#process post text and title to add more information (need to process separately to avoid duplicate text to cmt)

def all_vocab_extraction(df):
    # preprocessing the text column in dataframe

    df['comment_text'] = df.comment_text.apply(preprocessing)

    df_ = df[:]
    data_text = df_.comment_text.tolist()
    data_list = list(data_text)
    data_words = [i.split() for i in data_list]

    # pos_text = posta(data_words , allowed_postags=['NOUN', 'VERB'])

    df['pos_comment_text'] = data_words #pos_text 
    df['pos_comment_text'] = df['pos_comment_text'].apply(lambda x: ' '.join(map(str, x)))

    corpus =" "

    for i in range(len(df['pos_comment_text'])):
        corpus= corpus+ ' ' + df['pos_comment_text'].iloc[i]
        
    tokens_vs_stop = word_tokenize(corpus)
    all_text = [word for word in tokens_vs_stop]
    # total_vocabulary = [WordNetLemmatizer().lemmatize(word) for word in all_text]

    return all_text #total_vocabulary

def word_freq_bigram(text):
    
    _1gram = ''.join(text).split(" ")
    _2gram = [' '.join(e) for e in ngrams(_1gram, 2)]
#         _3gram = [' '.join(e) for e in ngrams(_1gram, 3)]
    word_dist = nltk.FreqDist(_2gram)
    most_freq_biGram = word_dist.most_common(50)
        
    return word_dist

def show_topics(vectorizer, lda_model, n_words=10):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords