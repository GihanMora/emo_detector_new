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
    lab= list(set(pred))+['un']
    print('labels_c_mat',lab)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all,labels=lab)
    acc = accuracy_score(labels_all, preds_all)
    confusion_mat = confusion_matrix(labels_all, preds_all,labels=lab)

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

# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\ISEAR_kwd_imp.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_go_emo_model.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_go_emo_model_cuz.csv")
ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\sem18_prediction_simple_kwd.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_semeval2018_go_emo_model_lmn_go_voc.csv")
ev_df = ev_df.dropna()


print(ev_df.columns)
print(ev_df['labels1'].unique())
y_ground = []
y_pred = []

# print(ev_df['sentiment'].unique())
# for i,row in ev_df.iterrows():
#   print(row)
#   print(row['sentiment_id'],int(row['sentiment_id']))
_list = []
for i, row in ev_df.iterrows():
    # print(row)
    ground_t = row['labels1']


    pred = row['predictions']
    # print(pred)
    pred = ast.literal_eval(pred.replace('Counter', ''))
    # pred = ft_to_et(pred)
    pred_out = {k: v for k, v in sorted(pred.items(), key=lambda x: x[1])}
    # print(pred)

    pred = list(pred_out.keys())[-1]
    print(pred,pred_out[pred])




    if (pred in ['fear','joy','sad','anger']):
        pred = list(pred_out.keys())[-2]
        if (pred in ['fear','joy','sad','anger']):
            pred = list(pred_out.keys())[-3]
            if (pred in ['fear', 'joy', 'sad', 'anger']):
                pred = list(pred_out.keys())[-4]
                if (pred in ['fear', 'joy', 'sad', 'anger']):
                    pred = list(pred_out.keys())[-5]


    #ISEAR


    if (pred in ['trust']):
        pred_id = 'trust'

    elif (pred in ['anticipation']):
        # pred_id = 4
        pred_id = 'anticipation'
    if (pred in ['amazement_surprise','surprise']):
        # pred_id = 3
        pred_id = 'surprise'
    elif (pred in ['disgust_loathing','disgust']):
        # pred_id = 3#ang_dis joined
        # pred_id = 5
        pred_id = 'disgust'

    if (pred_out[pred] == 0):
        pred = 'unknown'
        pred_id = 'un'
        print('not detected')

    if(pred_id!=ground_t):
        ground_t = row['labels2']



    y_ground.append(ground_t)
    y_pred.append(pred_id)
    # if (ground_t != pred_id):
    #     _list.append(row)


compute_metrics(y_ground, y_pred)

