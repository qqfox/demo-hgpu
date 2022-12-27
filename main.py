# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 18:35:11 2022

@author: Quinn
"""

import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
import string
import re
import gensim
import nltk
nltk.download('punkt')
from gensim.parsing.preprocessing import STOPWORDS
lemmatizer=WordNetLemmatizer()
seed = 42
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
import pickle
import warnings
warnings.filterwarnings("ignore")
import math
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from wordcloud import WordCloud
import streamlit as st
from streamlit import components

from nltk.util import ngrams
import pyLDAvis
import pyLDAvis.sklearn
# preprocessing part

nltk.download('punkt')
from gensim.parsing.preprocessing import STOPWORDS

from preprocessing_data import all_vocab_extraction, word_freq_bigram, preprocessing

header = st.container()
dataset = st.container()
features = st.container()
frequences = st.container()
lda_model = st.container()

@st.cache 
def get_data(file_name):  
    data = pd.read_csv(file_name)
    return data


with header:
    st.title('Haloo, welcome to Hanxun')
    st.text('This is a demo for GPU project userface')

with dataset:
    st.header('this dataset contains all the text crawled from reddit')
    st.text('dataset description')
    filename = r"C:\Users\USER\Documents\GitHub\demo-hgpu\data_rx6900_Reddit_070822.csv"
    text_data = pd.read_csv(filename, index_col = 0)

    # text_data = pd.read_csv(r'C:\Users\USER\Documents\GitHub\GPU\streamlit\data\data_rx6900_Reddit_070822.csv', index_col=0)
    st.write(text_data.head(5))

    st.text('Separate fist as we need to delete the duplication from posts data. later merge the comment and post text')
    st.text('comment dataframe columns') 
    st.write(text_data.columns)


with frequences:
    st.header('Time to train the model')
    st.text('Here you need to change to parameter of the model, and see if the results are good to capture')

    # text_data['comment_text'] = text_data.comment_text.apply(preprocessing(input_more_sw))
    text_data['comment_text'] = text_data.comment_text.apply(preprocessing)
    total_vocabulary = all_vocab_extraction(text_data)

    freq_dist = nltk.FreqDist(total_vocabulary)
    fig, ax = plt.subplots(figsize=(15,7))
    freq_dist.plot(25)
    st.pyplot(fig)

    st.text('frequently mentioned 2 words')

    fig2, ax = plt.subplots(figsize=(15,7))
    total_vocab_str = " ".join(total_vocabulary)
    topword = word_freq_bigram(total_vocab_str)
    topword.plot(25)
    st.pyplot(fig2)

    wordcloud = WordCloud(
            background_color='black',
            stopwords=stopwords,
            max_words=100,
            max_font_size=40, 
            scale=3,
            random_state=1) # chosen at random by flipping a coin; it was heads
    wordcloud.generate_from_frequencies(frequencies=freq_dist)
    fig3 = plt.figure(1, figsize=(20, 16))
    plt.axis('off')
    plt.imshow(wordcloud)
    # plt.savefig('demo_rx6900_Task3_cloud.png', dpi = 900)
    plt.show()
    st.pyplot(fig3)


with features:

    st.header('the features I created')
    st.markdown('* **Top frequent words** *')
    st.markdown('* process text from comments and create features *')



with lda_model:
    @st.cache

    def show_topics(vectorizer, lda_model, n_words): #num_words
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    sel_col, disp_col = st.columns(2)

    input_more_sw = sel_col.text_input('Please add in words that you want to remove in lower form, for examples', '"hello", "hi"')

    text = [word for word in total_vocabulary if not word in input_more_sw]
    
    num_words = sel_col.slider('choose the number of frequent words:', min_value=10, max_value=50,value=10,step=2)
    num_clusters = sel_col.selectbox('choose the number of clusters', options=[3,4,5,6,7,8],index=0)
    min_occur = sel_col.selectbox('set the minimum times that words occur in the corpus', options=[3,4,5,6,7,8,9,10],index=2)
    

    CountVec = CountVectorizer(max_df=0.95, min_df=min_occur, max_features=50000) #min_occur
    data_vectorized = CountVec.fit_transform(text) # fit input data
    lda_model_ = LatentDirichletAllocation(n_components=num_clusters, #num_clusters
                                        max_iter=10, 
                                        learning_method='online',
                                        learning_offset=70.,
                                        learning_decay = .7,
                                        random_state=0).fit(data_vectorized)
    topic_keywords = show_topics(CountVec, lda_model_, num_words)    #num_words 10    # number of topic kw can be changed

    disp_col.write(topic_keywords)

    
    html_ = pyLDAvis.sklearn.prepare(lda_model_, data_vectorized, CountVec)
    html_string = pyLDAvis.prepared_data_to_html(html_)
    components.v1.html(html_string, width=1300, height=800, scrolling=True)

