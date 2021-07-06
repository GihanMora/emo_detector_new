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
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_affective_text_refined.csv")
# ev_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\evaluations\affective_text_prediction_kwd_imp.csv")
ev_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\evaluations\affective_text_prediction_simple.csv")





# print(ev_df.columns)
y_ground = []
y_pred = []

# print(ev_df['sentiment'].unique())
# for i,row in ev_df.iterrows():
#   print(row)
#   print(row['sentiment_id'],int(row['sentiment_id']))
_list = []
for i, row in ev_df.iterrows():

    # ground_t = row['sentiment']
    # affective_text
    ground_t = row['sentiment_id']

    if(ground_t==5):continue
    if (ground_t == 1): continue
    pred = row['predictions']
    # print(pred)
    pred = ast.literal_eval(pred.replace('Counter', ''))
    # pred = ft_to_et(pred)
    pred_out = {k: v for k, v in sorted(pred.items(), key=lambda x: x[1])}
    # print(pred)

    pred = list(pred_out.keys())[-1]
    print(pred)
    if (pred_out[pred] == 0):
        pred_id = 5
        print('not detected')



    #affective text
    if(pred=='trust'):
        pred = list(pred_out.keys())[-2]
        if (pred == 'trust'):
            pred = list(pred_out.keys())[-3]

    affective_text_emo_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness', 5: 'surprise'}

    # print(pred)
    if (pred in ['acceptance', 'joy_ecstasy', 'joy', 'admire', 'senerity', 'trust', 'anticipation',
                 'interest_vigilance', 'amazement_surprise']):
        pred_id = 3
    elif (pred in ['fear']):
        pred_id = 2
    elif (pred in ['anger', 'distraction','disgust_loathing','disgust']):
        pred_id = 0
    elif (pred in ['sadness', 'sad', 'boredom']):
        pred_id = 4
    # elif (pred in ['disgust_loathing', 'disgust']):
    #     pred_id = 1
    # elif (pred in ['surprise']):
    #     pred_id = 5

    y_ground.append(ground_t)
    y_pred.append(pred_id)
    if (ground_t != pred_id):
        _list.append(row)
print(ground_t)
print(y_pred)

compute_metrics(y_ground, y_pred)



ffdd = pd.DataFrame(_list)

# ffdd.to_csv(r"E:\Projects\emo_detector_new\predictions\isear_14_false.csv")



# [[930 210 146 182 163]
#  [ 24 509 138  39 134]
#  [ 19  31 408  53 366]
#  [ 63 120 214 629 173]
#  [ 58 227 190 195 261]]