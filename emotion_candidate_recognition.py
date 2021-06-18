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

from get_nearest_neighbours import get_nearest_neighbours
from negation_handling import map_opposite_emotions, negations
from scoring import calculate_scores

# df = pd.read_csv(r'E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv')
# # print(df.head())
# # print(df.columns)
# df = df.dropna()


tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\results\checkpoint-2000")





#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # print('ime',input_mask_expanded)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # print('se',sum_embeddings)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_mean_pooling_emb(sentences):

    # tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
    # model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\results_goemotions\checkpoint-3395")
    # tokenizer = AutoTokenizer.from_pretrained(r"E:\Projects\emo_detector_new\go_model_simple")
    # model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\go_model_simple")
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings_raw = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = sentence_embeddings_raw.tolist()

    return sentence_embeddings



def emotion_candidates_recognition(sentence ,window_size,df):

    sentence_tokens = sentence.split(' ')
    # print(sentence_tokens)
    sentence_pieces = [sentence]
    for i in range(0 ,len(sentence_tokens ) + 1 -window_size):
        sliding_piece = ' '.join(sentence_tokens[i: i +window_size])
        # print(sliding_piece)
        sentence_pieces.append(sliding_piece)

    # print(sentence_pieces)
    sentence_emb = get_mean_pooling_emb(sentence_pieces)

    normalized_score_dict = get_nearest_neighbours([sentence_emb[0]],df)

    need_negation_check = False
    for each_ww in sentence_tokens:
        if(each_ww.strip().lower() in negations):
            need_negation_check = True
            break
    if(need_negation_check):
        tuples = []
        for i in range(1 ,len(sentence_emb)):
            sliding_piece = sentence_pieces[i]
            dis = cosine_similarity([sentence_emb[i]], [sentence_emb[0]])
            # print(dis)
            tuples.append([sliding_piece ,dis])
        # print([i[0] for i in tuples])
        # print([i[1].tolist()[0] for i in tuples])

        s_tup = sorted(tuples, key=lambda x: x[1]  )  # sort tuples based on the cosine distance
        # print(s_tup)

        top_windows = [i[0] for i in s_tup[::-1][:5]]

        if(window_size==1):

            # upper_threshold = int(0.33*len(sentence_tokens))
            upper_threshold = 1
            top_windows = [i[0] for i in s_tup[::-1][:upper_threshold]]
            fixed_top_windows = []
            for emoWord in top_windows:

                end_ind_int = sentence_tokens.index(emoWord)
                start_ind_int = end_ind_int - 3
                if start_ind_int < 0:
                    start_ind_int = 0
                text_chunk_int = (' ').join(sentence_tokens[start_ind_int:end_ind_int])
                fixed_top_windows.append(text_chunk_int.strip().lower())
            print('fixed',fixed_top_windows)

            top_windows = fixed_top_windows
        # print('top windows', top_windows)
        if(check_for_negations(top_windows)):
            print('Emotions are negated')
            normalized_score_dict = map_opposite_emotions(normalized_score_dict)





    # for s_t in s_tup[::-1]:

    #   print(s_t)


    # plot_emotional_weight([i[0] for i in tuples],[i[1].tolist()[0] for i in tuples])

    normalized_score_dict = {k: v for k, v in sorted(normalized_score_dict.items(), key=lambda item: item[1])}

    return normalized_score_dict

def plot_emotional_weight(x ,y):
    # plotting the points
    plt.figure(figsize=(20 ,5))
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('word windows')
    # naming the y axis
    plt.ylabel('emotional contribution')

    # giving a title to my graph
    # plt.title('Emotion contribution o')

    # function to show the plot

    plt.show()

def check_for_negations(top_candidates):
    neg = False
    for tsp in top_candidates:
        for each_p in tsp.split(' '):
            if(each_p in negations):
                # print('profiles are negated')
                neg = True
    return neg


# print(emotion_candidates_recognition('He is a smart person' ,1))