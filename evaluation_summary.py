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


# ev_df = pd.read_csv(r"/content/drive/MyDrive/NLP_notebooks/Emotional_embeddings/predictions_twitter_sentiment_new.csv")
ev_df = pd.read_csv(r"/content/drive/MyDrive/NLP_notebooks/Emotional_embeddings/predictions_new_dataset.csv")
# ev_df = pd.read_csv(r"/content/drive/MyDrive/NLP_notebooks/Emotional_embeddings/predictions_ISEAR_sentiment_all.csv")
# ev_df = pd.read_csv(r"/content/drive/MyDrive/NLP_notebooks/Emotional_embeddings/predictions_ISEAR_sentiment_4.csv")

print(ev_df.columns)
y_ground = []
y_pred = []
# for i,row in ev_df.iterrows():
#   print(row)
#   print(row['sentiment_id'],int(row['sentiment_id']))
_list = []
for i, row in ev_df.iterrows():
    # print(row)
    ground_t = int(row['sentiment_id'])
    if (ground_t != 1):
        ground_t = 0
    # if(ground_t==2):continue
    pred = row['predictions']
    # print(pred)
    pred = ast.literal_eval(pred.replace('Counter', ''))
    # print(pred)
    pred = list(pred.keys())[0]
    # print(pred)
    # if(pred in ['joy','trust','surprise','anticipation']):
    #   pred_id = 1
    # elif(pred=='fear'):
    #   pred_id = 2
    # elif(pred=='anger'):
    #   pred_id = 3
    # elif(pred=='sadness'):
    #   pred_id = 4
    # elif(pred=='disgust'):
    #   pred_id = 5

    if (pred in ['joy', 'trust', 'surprise', 'anticipation']):
        pred_id = 1
    elif (pred == 'fear'):
        pred_id = 0
    elif (pred == 'anger'):
        pred_id = 0
    elif (pred == 'sadness'):
        pred_id = 0
    elif (pred == 'disgust'):
        pred_id = 0

    y_ground.append(ground_t)
    y_pred.append(pred_id)
    if (ground_t != pred_id):
        _list.append(row)
print(ground_t)
print(y_pred)

compute_metrics(y_ground, y_pred)