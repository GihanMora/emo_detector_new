import numpy
import pandas as pd
import ast
from datetime import datetime
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModel, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import re               # Obtain expressions
from gensim.models import Word2Vec    #Import gensim Word2Fec
from sklearn.decomposition import PCA #Grab PCA functions
import numpy as np
#Plot helpers
import matplotlib
import matplotlib.pyplot as plt
#Enable matplotlib to be interactive (zoom etc)
import ast

from scoring import calculate_scores

# df = pd.read_csv(r'E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab.csv')

# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\go_emo_simple_new_vocab.csv")
# print(df.head())
# print(df.columns)

# print(df['fourteen_label'].unique())
# print(len(df))

# df = df.loc[df['fourteen_label'] != 'boredom']

# print(len(df))
from scipy import spatial


def get_nearest_neighbours(embeding,df):
    t1 = datetime.now()
    tuples = []

    for i, row_e in df.iterrows():
        # p = [ast.literal_eval(row_e['embedding'])]
        dis = cosine_similarity([row_e['embedding']], embeding)
        # dis = 1 - spatial.distance.cosine(embeding, embeding)
        # print([row_e['tokens'],row_d['tokens'],dis])
        tuples.append([row_e['token'], row_e['fourteen_label'], dis, row_e['embedding']])
    t2 = datetime.now()
    diff = t2 - t1
    print('time',diff)
    s_tup = sorted(tuples, key=lambda x: x[2])  # sort tuples based on the cosine distance
    neaarest_neighbs_words = []
    neaarest_neighbs_embs = []
    neaarest_neighbs_labels = []
    for i, m in enumerate(s_tup[::-1]):
        # print(m)
        if (i < 50):  # getting the nearest 100 neighbours
            neaarest_neighbs_words.append(m[0])
            neaarest_neighbs_embs.append(m[3])
            neaarest_neighbs_labels.append(m[1])
    n_score_dict = calculate_scores(neaarest_neighbs_words, neaarest_neighbs_labels)
    # print(numpy.shape(numpy.array(embeding[0])))
    # print(numpy.shape(neaarest_neighbs_embs[0]))
    neaarest_neighbs_words.append('sentence')
    neaarest_neighbs_embs.append(numpy.array(embeding[0]))
    neaarest_neighbs_labels.append('input')

    print(Counter(neaarest_neighbs_labels))
    # print(neaarest_neighbs_words)
    # print(neaarest_neighbs_labels)
    # for neighbour,label in zip(neaarest_neighbs_words,neaarest_neighbs_labels):
    #   print(neighbour,label)
    # print(neaarest_neighbs_embs)

    # visualize the neighbours in 2d space
    # visualize_embs(neaarest_neighbs_labels,neaarest_neighbs_embs,neaarest_neighbs_words)
    return n_score_dict