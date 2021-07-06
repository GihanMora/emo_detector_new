import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix

from evaluation_summary_fairy_tales import ft_to_et

emo_go_emotions = {
1:'admiration',
2:'amusement',
3:'anger',
4:'annoyance',
# 5:'trust',
5:'approval',
6:'caring',
7:'confusion',
8:'curiosity',
9:'desire',
10:'disappointment',
11:'disapproval',
12:'disgust',
13:'embarrassment',
14:'excitement',
15:'fear',
16:'gratitude',
17:'grief',
18:'joy',
19:'love',
20:'nervousness',
21:'optimism',
22:'pride',
23:'realization',
24:'relief',
25:'remorse',
26:'sadness',
27:'surprise',
28:'neutral'}



def compute_metrics(pred, ground_labels):
    labels_all = ground_labels
    preds_all = list(pred)

    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all)
    acc = accuracy_score(labels_all, preds_all)
    confusion_mat = confusion_matrix(labels_all, preds_all)

    out_dict = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusiton_mat': confusion_mat
    }
    for k in out_dict.keys():
        print(k)
        print(out_dict[k])

import pandas as pd
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_goemotions_refined.csv")
ev_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\evaluations\goemo_prediction_kwd_imp.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_goemotions_dual_bert.csv")
# print(ev_df.columns)
y_ground = []
y_pred = []

# print(ev_df['sentiment'].unique())
# for i,row in ev_df.iterrows():
#   print(row)
#   print(row['sentiment_id'],int(row['sentiment_id']))
_list = []
for i, row in ev_df.iterrows():

    ground_t = row['sentiment']
    # ground_t = int(ground_t.split(',')[0]) + 1

    # go emotions
    ground_t = int(ground_t.split(',')[0])+1
    if (emo_go_emotions[ground_t] == 'joy'):
        ground_t = 1
    elif (emo_go_emotions[ground_t]== 'anger'):
        ground_t = 2
    # elif (emo_go_emotions[ground_t] == 'trust'):
    #     ground_t = 3
    elif (emo_go_emotions[ground_t] == 'surprise'):
        ground_t = 4
    # elif (emo_go_emotions[ground_t] == 'anticipation'):
    #     ground_t = 5
    elif (emo_go_emotions[ground_t] == 'fear'):
        ground_t = 6
    elif (emo_go_emotions[ground_t] in ['sad','sadness']):
        ground_t = 7
    # elif (emo_go_emotions[ground_t] == 'disgust'):
    #     ground_t = 8
    # elif (emo_go_emotions[ground_t] == 'annoyance'):
    #     ground_t = 9
    # elif (emo_go_emotions[ground_t] == 'admiration'):
    #     ground_t = 10
    # elif (emo_go_emotions[ground_t] == 'confusion'):
    #     ground_t = 11
    else:
        continue

    pred = row['predictions']
    # print(pred)
    pred = ast.literal_eval(pred.replace('Counter', ''))
    # pred = ft_to_et(pred)
    pred_out = {k: v for k, v in sorted(pred.items(), key=lambda x: x[1])}
    # print(pred)

    pred = list(pred_out.keys())[-1]
    print(pred)
    # pred = emo_go_emotions[pred+1]
    # print(pred)


    #go emotions
    our_emo_map = {
        'joy':1,
        'joy_ecstasy': 1,
        'senerity':1,
        'anger':2,
        # 'trust':3,
        'surprise':4,
        'amazement_surprise':4,
        'distraction': 4,
        # 'anticipation':5,
        # 'interest_vigilance':5,
        'fear':6,
        'sadness':7,
        'sad':7,
        # 'disgust':8,
        # 'disgust_loathing':8,
        # 'boredom':2,
        # 'admire':3,
        # 'acceptance':3

    }
    #
    if (pred in ['anticipation','boredom','trust','interest_vigilance','admire','disgust','acceptance','disgust_loathing']):
        pred = list(pred_out.keys())[-2]
        if (pred in ['anticipation','boredom','trust','interest_vigilance','admire','disgust','acceptance','disgust_loathing']):
            pred = list(pred_out.keys())[-3]
            if (pred in ['anticipation','boredom','trust','interest_vigilance','admire','disgust','acceptance','disgust_loathing']):
                continue
    pred_id = our_emo_map[pred]


    y_ground.append(ground_t)
    y_pred.append(pred_id)
    if (ground_t != pred_id):
        _list.append(row)
print(ground_t)
print(y_pred)

compute_metrics(y_ground, y_pred)



