import ast

import nltk
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from get_nearest_neighbours import get_nearest_neighbours

nltk.download('punkt')
from emotion_candidate_recognition import emotion_candidates_recognition_lnm, emotion_candidates_recognition, \
    get_mean_pooling_emb


def emo_detecct_document(text):
    tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\emo_bert_model")
    model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\emo_bert_model")
    df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv")
    df = df.dropna()
    df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]
    output_emo_dict = {
              'anticipation':0,
              'anger':0,
              'fear':0,
              'sadness':0,
              'trust':0,
              'senerity':0,
              'joy_ecstasy':0,
              'admire':0,
              'acceptance':0,
              'amazement_surprise':0,
              'distraction':0,
              'boredom':0,
              'disgust_loathing':0,
              'interest_vigilance':0}
    a_list = nltk.tokenize.sent_tokenize(text)

    for each_s in a_list:
        print(each_s)
        pred = emotion_candidates_recognition(each_s,1,df,tokenizer,model)
        for each_k in pred.keys():
            output_emo_dict[each_k]=output_emo_dict[each_k]+pred[each_k]

    print(output_emo_dict)
    return output_emo_dict

from datetime import datetime
def emo_detect_document_lnm(text,tokenizer,model,step,df):
    tt = datetime.now()
    output_emo_dict = {
              'negative':0,
              'positive':0,
              'uncertainty':0,
              'litigious':0,
              'model_strong':0,
              'model_weak':0,
              }
    tm = datetime.now()
    a_list = nltk.tokenize.sent_tokenize(text)
    a_list = [(' ').join(a_list[i:i + step]) for i in range(0, len(a_list), step)]
    sentence_embs = get_mean_pooling_emb(a_list, tokenizer, model)
    ttm = datetime.now()
    dis = ttm-tm
    print('emb_ex',dis)
    print(len(sentence_embs))
    import numpy as np
    print('np_shape', np.shape(sentence_embs))

    # for each_s in a_list:
    for sentence_emb in sentence_embs:
        # print(each_s)
        # pred = emotion_candidates_recognition(each_s,1,df,tokenizer,model)
        pred = get_nearest_neighbours([sentence_emb], df)
        for each_k in pred.keys():
            output_emo_dict[each_k] = output_emo_dict[each_k] + pred[each_k]
    # for each_s in a_list:
    #     print(each_s)
    #     ttu = datetime.now()
    #     pred = emotion_candidates_recognition_lnm(each_s,1,df,tokenizer,model)
    #     ttu2 = datetime.now()
    #     difffu = ttu2 - ttu
    #     print('per pass', difffu)
    #     for each_k in pred.keys():
    #         output_emo_dict[each_k]=output_emo_dict[each_k]+pred[each_k]

    print(output_emo_dict)
    tt2 = datetime.now()
    difff = tt2 - tt
    print('per doc', difff)
    return output_emo_dict


# tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\emo_bert_model")
# model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\emo_bert_model")
# text = ''
# emo_detect_document_lnm(text,tokenizer,model)
# print(pred)
#
# s = ['a','b','c','d','e','d']
# b = [(' ').join(s[i:i+2]) for i in range(0,len(s),2)]
# print(b)