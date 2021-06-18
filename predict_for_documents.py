import ast

import nltk
import pandas as pd
nltk.download('punkt')
from emotion_candidate_recognition import emotion_candidates_recognition



def emo_detecct_document(text):
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
        pred = emotion_candidates_recognition(each_s,1,df)
        for each_k in pred.keys():
            output_emo_dict[each_k]=output_emo_dict[each_k]+pred[each_k]

    print(output_emo_dict)
    return output_emo_dict
text = ''
emo_detecct_document(text)
# print(pred)