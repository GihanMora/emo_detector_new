import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix


def ft_to_et(emo_dict):

    eight_dict = {'sadness': 0, 'anger': 0, 'joy': 0, 'anticipation': 0, 'fear': 0, 'surprise': 0, 'disgust': 0,
                  'trust': 0}

    if ('anticipation' in emo_dict.keys()):
        eight_dict['anticipation'] += emo_dict['anticipation']
    if ('amazement_surprise' in emo_dict.keys()):
        eight_dict['surprise'] += emo_dict['amazement_surprise']
    if ('distraction' in emo_dict.keys()):
        eight_dict['surprise'] += emo_dict['distraction']
    if ('fear' in emo_dict.keys()):
        eight_dict['fear'] += emo_dict['fear']
    if ('trust' in emo_dict.keys()):
        eight_dict['trust'] += emo_dict['trust']
    if ('interest_vigilance' in emo_dict.keys()):
        eight_dict['anticipation'] += emo_dict['interest_vigilance']
    if ('anger' in emo_dict.keys()):
        eight_dict['anger'] += emo_dict['anger']
    if ('disgust_loathing' in emo_dict.keys()):
        eight_dict['disgust'] += emo_dict['disgust_loathing']
    if ('senerity' in emo_dict.keys()):
        eight_dict['joy'] += emo_dict['senerity']
    if ('boredom' in emo_dict.keys()):
        eight_dict['disgust'] += emo_dict['boredom']
    if ('acceptance' in emo_dict.keys()):
        eight_dict['trust'] += emo_dict['acceptance']
    if ('joy_ecstasy' in emo_dict.keys()):
        eight_dict['joy'] += emo_dict['joy_ecstasy']
    if ('admire' in emo_dict.keys()):
        eight_dict['trust'] += emo_dict['admire']
    if ('sadness' in emo_dict.keys()):
        eight_dict['sadness'] += emo_dict['sadness']

    # print(eight_dict)
    return eight_dict


import pandas as pd
path = r"E:\Projects\emo_detector_new\predictions\fairy_tales_curated_neg_bos_inh.csv"


# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_affective_text_refined.csv")
# ev_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\evaluations\affective_text_prediction_kwd_imp.csv")
ev_df = pd.read_csv(path)
print(len(ev_df))
_list = []
tls = []
ns_i = []
for i, row in ev_df.iterrows():
    # emo_dict_fairy_tale = {2: 'Angry-Disgusted', 3: 'Fearful', 4: 'Happy', 6: 'Sad', 7: 'Surprised'}
    # tls.append(len(row['sentiment']))
    print(row['sentiment'])
    senti = row['sentiment']
    if(senti=='Sad'):
        senti_neg = 'Happy'
        nsi = 4
    elif (senti == 'Happy'):
        senti_neg = 'Sad'
        nsi = 6
    elif (senti == 'Angry-Disgusted'):
        senti_neg = 'Happy'
        nsi = 4
    elif (senti == 'Fearful'):
        senti_neg = 'Happy'
        nsi = 4

    tls.append(senti_neg)
    ns_i.append(nsi)

# ffdd = pd.DataFrame(_list)
ev_df['senti_negated'] = tls
ev_df['senti_neg_id'] = ns_i
ev_df.to_csv(r"E:\Projects\emo_detector_new\predictions\fairy_tales_cuared_final.csv")



# [[930 210 146 182 163]
#  [ 24 509 138  39 134]
#  [ 19  31 408  53 366]
#  [ 63 120 214 629 173]
#  [ 58 227 190 195 261]]