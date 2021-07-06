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


def map_candidate_to_emotions(neighbour_dict, candidate_dict):
    # print(neighbour_dict)
    # print(len(neighbour_dict['words']))
    # print(candidate_dict)
    emo_candi_dict = {}
    neighbor_df = pd.DataFrame(neighbour_dict)
    # print(neighbor_df.head())

    dft = neighbor_df.groupby('labels')['words'].nunique().sort_values(ascending=False).reset_index(name='count')
    unique_emos = dft['labels'][:3]
    # print(unique_emos)

    for each_cd in candidate_dict:
        dis_emo_dict = {}
        for each_ue in unique_emos:
            dis_list = []
            emod = neighbor_df.loc[neighbor_df['labels'] == each_ue][:50]
            for j, e_row in emod.iterrows():
                dis_list.append(cosine_similarity([e_row['embs']], [each_cd[2]]))
            # print(np.mean(dis_list))
            dis_emo_dict[each_ue] = np.mean(dis_list)
        print(each_cd[0])
        # print(dis_emo_dict)
        print(max(dis_emo_dict, key=dis_emo_dict.get))
        emo_candi_dict[each_cd[0]]: max(dis_emo_dict, key=dis_emo_dict.get)
        # break
    return emo_candi_dict





#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # print('ime',input_mask_expanded)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # print('se',sum_embeddings)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_mean_pooling_emb(sentences,tokenizer,model):

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



def emotion_candidates_recognition(sentence ,window_size,df,tokenizer,model):

    sentence_tokens = sentence.split(' ')
    # print(sentence_tokens)
    sentence_pieces = [sentence]
    for i in range(0 ,len(sentence_tokens ) + 1 -window_size):
        sliding_piece = ' '.join(sentence_tokens[i: i +window_size])
        # print(sliding_piece)
        sentence_pieces.append(sliding_piece)

    # print(sentence_pieces)
    sentence_emb = get_mean_pooling_emb(sentence_pieces,tokenizer,model)

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

def emotion_candidates_recognition_lnm(sentence ,window_size,df,tokenizer,model):

    sentence_tokens = sentence.split(' ')
    # print(sentence_tokens)
    sentence_pieces = [sentence]
    for i in range(0 ,len(sentence_tokens ) + 1 -window_size):
        sliding_piece = ' '.join(sentence_tokens[i: i +window_size])
        # print(sliding_piece)
        sentence_pieces.append(sliding_piece)

    # print(sentence_pieces)
    sentence_emb = get_mean_pooling_emb(sentence_pieces,tokenizer,model)

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

def sum_up_dicts(emo_dicts):
    sum_dict = {
        'negative': 0,
        'positive': 0,
        'uncertainty': 0,
        'litigious': 0,
        'model_strong': 0,
        'model_weak': 0,
        'anticipation': 0,
        'anger': 0,
        'fear': 0,
        'sadness': 0,
        'trust': 0,
        'senerity': 0,
        'joy_ecstasy': 0,
        'joy': 0,
        'sad': 0,
        'admire': 0,
        'acceptance': 0,
        'amazement_surprise': 0,
        'surprise': 0,
        'distraction': 0,
        'boredom': 0,
        'disgust_loathing': 0,
        'disgust': 0,
        'interest_vigilance': 0}

    for each_dict in emo_dicts:
        for each_k in each_dict.keys():
            sum_dict[each_k] = sum_dict[each_k] + each_dict[each_k]
    final_sum_dict = sum_dict.copy()
    for k in sum_dict.keys():
        if sum_dict[k] == 0:
            del final_sum_dict[k]
    print(final_sum_dict)
    return final_sum_dict

# sum_up_dicts([{'negative': 5, 'positive': 0.125, 'uncertainty': 0.283, 'litigious': 0.122, 'model_strong': 0.05, 'model_weak': 0.039},
#               {'negative': 0.422, 'positive': 0.125, 'uncertainty': 0.283, 'litigious': 3, 'model_strong': 0.05, 'model_weak': 0.039}])



import ast
tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\emo_bert_model")
model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\emo_bert_model")


df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv")
df = df.dropna()
df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]

print(emotion_candidates_recognition('Chefs not counting calories, study finds' ,1,df,tokenizer,model))

# {'fear': 0.022, 'surprise': 0.027, 'anticipation': 0.027, 'trust': 0.038, 'senerity': 0.038, 'distraction': 0.044, 'interest_vigilance': 0.087, 'boredom': 0.1, 'joy': 0.199, 'disgust': 0.211, 'sadness': 0.248}