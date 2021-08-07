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


ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_lnm_model_financial_phrasebank_even_cuz.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions/predictions_financialpb_go_emo_model.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions/predictions_financialpb_go_emo_model_lmn_go_voc.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_lnm_model_financial_phrasebank_even_cuz.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\financial_pb_goemo_with_keys_joy_sad.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\goemo_online_financial_pb.csv")

# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions/predictions_lnm_model_fpb_lnm_equal_pn.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions/predictions_lnm_model_fpb_lnm_equal_pn_extended_dd_1.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions/predictions_lnm_model_fpb_lnm_goemo_equal_pn_extended_dd.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions/predictions_fpb_go_emo_lnm_extended.csv")
# ev_df = pd.read_csv(r"E:\Projects\emo_detector_new\predictions/predictions_lnm_model_fpb_lnm_goemo_equal_pn_extended_dd_10_down.csv")
ev_df = ev_df.dropna()


print(ev_df.columns)
# print(ev_df['sentiment'].unique())
y_ground = []
y_pred = []
# print(ss)
# print(ev_df['sentiment'].unique())
# for i,row in ev_df.iterrows():
#   print(row)
#   print(row['sentiment_id'],int(row['sentiment_id']))
_list = []
for i, row in ev_df.iterrows():
    # print(row)
    ground_t = row['sentiment']
    # ground_t = row['g']


    # if (ground_t == 'positive'):
    #     ground_t = 1
    # if (ground_t == 'negative'):
    #     ground_t = 0
    if (ground_t == 'neutral'):
        continue



    pred = row['predictions']
    # pred = row['p']
    # print(pred)
    pred = ast.literal_eval(pred.replace('Counter', ''))
    # pred = ft_to_et(pred)

    print(pred)
    # {'positive': 0.034, 'uncertainty': 0.056, 'litigious': 0.068, 'model_strong': 0.085, 'negative': 0.798}
    # if ('litigious' in pred.keys()):
    #     try:
    #         pred['negative'] = pred['negative'] + pred['litigious']
    #     except:
    #         pred['positive'] = pred['positive'] + pred['litigious']
    # if ('uncertainty' in pred.keys()):
    #     try:
    #         pred['negative'] = pred['negative'] + pred['uncertainty']
    #     except:
    #         pred['positive'] = pred['positive'] + pred['uncertainty']
    # pos = ['positive', 'senerity','joy','acceptance','trust','amazement_surprise','surprise','anticipation','interest_vigilance']
    # neg = ['fear','negative', 'sad','boredom','disgust_loathing', 'disgust','distraction','anger']
    # for p in pos:
    #     if (p in pred.keys()):
    #         try:
    #             pred['joy_ecstasy'] = pred['joy_ecstasy'] + pred[p]
    #         except:
    #             pred['joy_ecstasy'] = 0
    #             pred['joy_ecstasy'] = pred['joy_ecstasy'] + pred[p]
    # for n in neg:
    #     if (n in pred.keys()):
    #         try:
    #             pred['sadness'] = pred['sadness'] + pred[n]
    #         except:
    #             pred['sadness'] = 0
    #             pred['sadness'] = pred['sadness'] + pred[n]



    # if('joy_ecstasy' in pred.keys()):
    #     pred['positive'] = pred['positive'] + pred['joy_ecstasy']
    # if ('sadness' in pred.keys()):
    #     pred['negative'] = pred['negative'] + pred['sadness']
    pred_out = {k: v for k, v in sorted(pred.items(), key=lambda x: x[1])}

    # pred_out = pred
    print(pred)

    pred = list(pred_out.keys())[-1]
    print(pred)
    # pred_id = pred


    # #ISEAR
    #
    #
    if (pred in ['joy_ecstasy','positive', 'senerity','joy','acceptance','trust','amazement_surprise','surprise','anticipation','interest_vigilance']):
        pred_id = 'positive'
    elif (pred in ['fear','negative']):
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

    # if (pred_out[pred] == 0):
    #     pred = 'unknown'
    #     pred_id = 10
    #     print('not detected')


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
    if (ground_t != pred_id):
        _list.append(row)


# print(ground_t)
# print(y_pred)

compute_metrics(y_ground, y_pred)



ffdd = pd.DataFrame(_list)

ffdd.to_csv(r"E:\Projects\emo_detector_new\predictions\financialpb_false_new_ext.csv")



# [[930 210 146 182 163]
#  [ 24 509 138  39 134]
#  [ 19  31 408  53 366]
#  [ 63 120 214 629 173]
#  [ 58 227 190 195 261]]