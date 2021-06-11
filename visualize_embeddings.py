

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




def visualize_embs(labels,embs,words):
    label_color_dict = {'input':'black','anticipation':'green','anger':'red','disgust':'red',
                        'fear':'red','sadness':'red','sad':'red','joy':'green','surprise':'green',
                        'trust':'green','senerity':'green',
                        'joy_ecstasy':'green',
                        'admire':'green',
                        'acceptance':'green',
                        'amazement_surprise':'green',
                        'distraction':'red',
                        'boredom':'red',
                        'disgust_loathing':'red',
                        'interest_vigilance':'green'}
    X = embs
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # print(len(result))
    # print(len(words))
    # print(len(labels))
    # print(len(embs))
    # filtered_words = []
    # filtered_emb = []
    # filtered_label = []
    # for i,j in enumerate(result):
    #     if(j[1] < -0.05 and labels[i] in ['joy','anticipation','trust','surprise']):
    #         filtered_emb.append(j)
    #         filtered_label.append(labels[i])
    #
    #         filtered_words.append(words[i])
    #
    #     if (j[1] > 0.05 and labels[i] in ['sad','disgust','anger','fear']):
    #         filtered_emb.append(j)
    #         filtered_label.append(labels[i])
    #         filtered_words.append(words[i])
    # result = np.array([x for x in filtered_emb])
    # labels = filtered_label

    cvec = [label_color_dict[label] for label in labels]
    # print('filtered words')
    # print(filtered_words)
    # print('filtered labels')
    # print(filtered_label)
    # fig, ax = plt.subplots()
    # ax.plot(result[:, 0], result[:, 1], 'o')
    # ax.set_title('Tweets')
    # plt.show()



    # Create the scatter plot
    plt.figure(figsize=(8,8))
    plt.scatter(result[:,0], result[:,1],c=cvec, edgecolor='', alpha=0.5)

    # selected_names = ['sad','joy']
    # names = ['sad','joy','surprise','trust','anticipation','anger','disgust','fear']
    # # Add the labels
    # for name in selected_names:
    #
    #     # Get the index of the name
    #     i = names.index(name)
    #
    #     # Add the text label
    #     labelpad = 0.01   # Adjust this based on your dataset
    #     plt.text(result[i,0]+labelpad, result[i,1]+labelpad, name, fontsize=9)
    #
    #     # Mark the labeled observations with a star marker
    #     plt.scatter(result[i,0], result[i,1],
    #                 c=cvec[i], vmin=min(cvec), vmax=max(cvec),
    #                 edgecolor='', marker='*', s=100)

    # Add the axis labels
    plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
    plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
    plt.title('all positive vs all negative')
    # Done
    plt.show()