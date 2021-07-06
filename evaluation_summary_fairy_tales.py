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


emo_go_emotions = {
1:'admiration',
2:'amusement',
3:'anger',
4:'annoyance',
# 5:'approval',
5:'trust',
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
# ev_df = pd.read_csv(r"/content/drive/MyDrive/NLP_notebooks/Emotional_embeddings/predictions_twitter_sentiment_new.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_sentiment_original_detector_all.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_sentiment_new_all.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_sentiment_new_14.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_fairy_tales_embert_all_14_refined.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_goemotions_new_14.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_affective_text_new_14.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_twitter.csv")
# ev_df = pd.read_csv(r"/content/drive/MyDrive/NLP_notebooks/Emotional_embeddings/predictions_ISEAR_sentiment_all.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_twitter_original_emo_detector.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_goemotions.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_fairy_tale_new_14.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_go_all_14.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\fairy_tales_with_keywords.csv")
# ev_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\evaluations\fairy_tales_prediction_kwd_imp.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\fairy_tales_full_pred.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\fairy_tales_cuared_final.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\fairy_tales_completed_new.csv")
# ev_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\evaluations/fairy_tales_prediction_simple.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\fairy_tales_kwd_imp.csv")


print(ev_df.columns)
y_ground = []
y_pred = []

# print(ev_df['sentiment'].unique())
# for i,row in ev_df.iterrows():
#   print(row)
#   print(row['sentiment_id'],int(row['sentiment_id']))
_list = []
for i, row in ev_df.iterrows():
    # print(row)
    # ground_t = int(row['sentiment_id'])
    # #goemotions
    # ground_t = row['sentiment']
    # affective_text
    ground_t = row['sentiment_id']
    # ground_t = row['senti_neg_id']
    # ground_t = row['ground']
    # if(ground_t!=1):
    #     ground_t=0
    # emo_dict_fairy_tale = {2: 'Angry-Disgusted', 3: 'Fearful', 4: 'Happy', 6: 'Sad', 7: 'Surprised'}
    if (ground_t == 2):
        ground_t = 1
    if (ground_t == 3):
        ground_t = 2
    if (ground_t == 4):
        ground_t = 3
    if (ground_t == 6):
        ground_t = 4
    if (ground_t == 7):
        # ground_t = 5
        continue
    # if (ground_t == 5):
    #     ground_t = 3
    #     continue

    # if (ground_t == 'positive'):
    #     ground_t = 1
    # if (ground_t == 'negative'):
    #     ground_t = 0
    # if (ground_t == 'neutral'):
    #     continue





    pred = row['predictions']
    # # pred = row['predictions_negated']
    #
    # # # print(pred)
    pred = ast.literal_eval(pred.replace('Counter', ''))
    # # pred = ft_to_et(pred)
    pred_out = {k: v for k, v in sorted(pred.items(), key=lambda x: x[1])}
    # # print(pred)
    #
    pred = list(pred_out.keys())[-1]
    # print(pred)

    if(pred_out[pred]==0):
        pred = 'unknown'
        pred_id = 10
        print('not detected')




    #fairy tales
    emo_dict_fairy_tale = {2: 'Angry-Disgusted', 3: 'Fearful', 4: 'Happy', 6: 'Sad', 7: 'Surprised'}
    if (pred in ['acceptance', 'joy_ecstasy', 'joy', 'admire', 'senerity', 'trust', 'anticipation',
                 'interest_vigilance','surprise','amazement_surprise']):
        pred_id = 3
    elif (pred in ['fear']):
        pred_id = 2
    elif (pred in ['sadness', 'sad','boredom','distraction']):
        pred_id = 4
    elif (pred in ['anger','disgust_loathing', 'disgust']):
        pred_id = 1
    # elif (pred in ['surprise','amazement_surprise']):
    #     pred_id = 5
    # # ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    # # pred = row['pred_emobert_neg']
    # # pred_id = int(pred)
    # # print(ground_t)
    # # if(pred_id==0):pred_id=4
    # # elif(pred_id==1):pred_id=3
    # # elif(pred_id==2):pred_id=10
    # # elif (pred_id == 3):
    # #     pred_id = 1
    # # elif (pred_id == 4):
    # #     pred_id = 2
    # # elif (pred_id == 5):
    # #     pred_id = 10


    # emobert
    # print(row['ground'], row['pred'])

    # pred = row['pred']
    # if (pred == 0):
    #     pred_id = 4
    # if (pred == 1):
    #     pred_id = 3
    # if (pred == 2):
    #     pred_id = 10
    # if (pred == 3):
    #     pred_id = 1
    # if (pred == 4):
    #     pred_id = 2
    # if (pred == 5):
    #     pred_id = 10

    y_ground.append(ground_t)
    y_pred.append(pred_id)
    # if (ground_t != pred_id):
    #     _list.append(row)
print('g',y_ground)
print('p',y_pred)

compute_metrics(y_ground, y_pred)



ffdd = pd.DataFrame(_list)
#
# ffdd.to_csv(r"E:\Projects\emo_detector_new\predictions\fairy_tales_incorrect.csv")
#


# [[930 210 146 182 163]
#  [ 24 509 138  39 134]
#  [ 19  31 408  53 366]
#  [ 63 120 214 629 173]
#  [ 58 227 190 195 261]]