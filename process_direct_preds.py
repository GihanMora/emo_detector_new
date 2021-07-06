import pandas as pd
import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix


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


dff= pd.read_csv(r"E:\Projects\emo_detector_new\predictions\direct_out_emo_bert_ISEAR.csv")
# dff = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\direct_out_emo_bert_affective_text.csv")
# dff = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\direct_out_emo_bert_fairy_tales.csv")
# dff = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\direct_out_emo_bert_twitter.csv")
direct_pred_dict = {0:'sadness',1: 'joy',2: 'love',3: 'anger',4: 'fear',5:'surprise'}
ISEAR_dict = {1:'joy',2:'fear',3:'anger',4:'sadness',5:'disgust'}



preds = []





#ISEAR
for i, row in dff.iterrows():
    pred_ = row['pred']
    if(pred_==0):
        pred_id = 4
    elif (pred_ == 1):
        pred_id = 1
    elif (pred_ == 2):
        pred_id = 10
    elif (pred_ == 3):
        pred_id = 3
    elif (pred_ == 4):
        pred_id = 2
    elif(pred_ == 5):
        pred_id = 10
    else:
        pred_id = pred_

    preds.append(pred_id)

#affective_text
# affective_text_emo_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness', 5: 'surprise'}
# direct_pred_dict = {0:'sadness',1: 'joy',2: 'love',3: 'anger',4: 'fear',5:'surprise'}
# for i, row in dff.iterrows():
#     pred_ = row['pred']
#     # print(pred_)
#     if(pred_==0):
#         pred_id = 4
#     if (pred_ == 1):
#         pred_id = 3
#     if (pred_ == 2):
#         pred_id = 10
#     if(pred_ == 3):
#         pred_id = 0
#     if (pred_ == 4):
#         pred_id = 2
#     if (pred_ == 5):
#         pred_id = 5
# #
# #     else:
# #         pred_id = pred_
# #
#     preds.append(pred_id)

#fairytales
# emo_dict_fairy_tale = {2: 'Angry-Disgusted', 3: 'Fearful', 4: 'Happy', 6: 'Sad', 7: 'Surprised'}
# direct_pred_dict = {0:'sadness',1: 'joy',2: 'love',3: 'anger',4: 'fear',5:'surprise'}
# for i, row in dff.iterrows():
#     pred_ = row['pred']
#     # print(pred_)
#     if(pred_==0):
#         pred_id = 6
#     if (pred_ == 1):
#         pred_id = 4
#     if (pred_ == 2):
#         pred_id = 10
#     if(pred_ == 3):
#         pred_id = 2
#     if (pred_ == 4):
#         pred_id = 3
#     if (pred_ == 5):
#         pred_id = 7
# #
# #     else:
# #         pred_id = pred_
# #
#     preds.append(pred_id)

#twitter
# direct_pred_dict = {0:'sadness',1: 'joy',2: 'love',3: 'anger',4: 'fear',5:'surprise'}
# gt = []
# for i, row in dff.iterrows():
#     pred_ = row['pred']
#     # print(pred_)
#     if(pred_ in [0,4]):
#         pred_id = 0
#     if (pred_ in [1,2,5]):
#         pred_id = 1
#     if(row['ground']==2):continue
#     else:gt.append(row['ground'])
#     preds.append(pred_id)
#
# compute_metrics(preds, gt)
compute_metrics(preds, dff['ground'])
