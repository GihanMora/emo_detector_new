import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix
emo_go_emotions = {
1:'admiration',
2:'amusement',
3:'anger',
4:'annoyance',
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
ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_goemo_bert.csv")

# print(ev_df.columns)
y_ground = []
y_pred = []

# print(ev_df['sentiment'].unique())
# for i,row in ev_df.iterrows():
#   print(row)
#   print(row['sentiment_id'],int(row['sentiment_id']))
_list = []
for i, row in ev_df.iterrows():

    ground_t = row['sentiment_id']
    pred = row['predictions']
    print(pred)
    pred = emo_go_emotions[pred+1]
    # print(pred)

    if(pred in ['joy','optimism','amusement']):
      pred_id = 1
    elif(pred in ['fear']):
      pred_id = 2
    elif(pred in ['anger','annoyance']):
      pred_id = 3
    elif(pred in ['sadness','grief','disappointment']):
      pred_id = 4
    elif(pred in ['disgust']):
      pred_id = 5
    # elif (pred == 'unknown'):
    else:
      pred_id = 6


    y_ground.append(ground_t)
    y_pred.append(pred_id)
    if (ground_t != pred_id):
        _list.append(row)
print(ground_t)
print(y_pred)

compute_metrics(y_ground, y_pred)



