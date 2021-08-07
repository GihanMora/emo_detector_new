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


ev_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\evaluations\twitter_prediction_simple.csv")
ev_df = ev_df.dropna()


print(ev_df.columns)
print(ev_df['sentiment'].unique())
y_ground = []
y_pred = []

# print(ev_df['sentiment'].unique())
# for i,row in ev_df.iterrows():
#   print(row)
#   print(row['sentiment_id'],int(row['sentiment_id']))
_list = []
for i, row in ev_df.iterrows():
    # print(row)
    ground_t = row['sentiment']


    # if (ground_t == 'positive'):
    #     ground_t = 1
    # if (ground_t == 'negative'):
    #     ground_t = 0
    if (ground_t == 'neutral'):
        continue



    pred = row['predictions']
    # print(pred)
    pred = ast.literal_eval(pred.replace('Counter', ''))
    # pred = ft_to_et(pred)
    pred_out = {k: v for k, v in sorted(pred.items(), key=lambda x: x[1])}
    # print(pred)

    pred = list(pred_out.keys())[-1]
    print(pred)


    #ISEAR


    if (pred in ['joy_ecstasy', 'senerity','joy','acceptance','trust','amazement_surprise','surprise','anticipation','interest_vigilance']):
        pred_id = 'positive'
    elif (pred in ['fear']):
        # pred_id = 2
        pred_id = 'negative'
    elif (pred in ['sadness', 'sad','boredom']):
        # pred_id = 4
        pred_id = 'negative'
    elif (pred in ['anger']):
        # pred_id = 3
        pred_id = 'negative'
    elif (pred in ['disgust_loathing', 'disgust','distraction',]):
        # pred_id = 3#ang_dis joined
        # pred_id = 5
        pred_id = 'negative'

    if (pred_out[pred] == 0):
        pred = 'unknown'
        pred_id = 10
        print('not detected')


    #emobert
    # print(row['ground'],row['pred'])
    # ground_t = row['ground']
    # pred = row['pred']
    # if(pred == 0):
    #     pred_id = 4
    # if (pred == 1):
    #     pred_id = 1
    # if (pred == 2):
    #     pred_id = 10
    # if (pred == 3):
    #     pred_id = 3
    # if (pred == 4):
    #     pred_id = 2
    # if (pred == 5):
    #     pred_id = 10
    # labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    y_ground.append(ground_t)
    y_pred.append(pred_id)
    # if (ground_t != pred_id):
    #     _list.append(row)


# print(ground_t)
# print(y_pred)

compute_metrics(y_ground, y_pred)



# ffdd = pd.DataFrame(_list)

# ffdd.to_csv(r"E:\Projects\emo_detector_new\predictions\isear_14_false.csv")



# [[930 210 146 182 163]
#  [ 24 509 138  39 134]
#  [ 19  31 408  53 366]
#  [ 63 120 214 629 173]
#  [ 58 227 190 195 261]]