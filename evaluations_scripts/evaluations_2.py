import ast

import pandas as pd
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/procesd_emotionlines_emo_ids.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/sentences_wc.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/twitter_dataset.csv"
from transformers import AutoTokenizer, AutoModel

from emotion_candidate_recognition import emotion_candidates_recognition, get_mean_pooling_emb

# tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\emo_bert_model")
# model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\emo_bert_model")
tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\goemo_emo_double_bert_model")
model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\goemo_emo_double_bert_model")


df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv")
df = df.dropna()
df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]



path = r"E:\Projects\emo_detector_new\datasets/goemotions.csv"
path = r""
results_df = pd.DataFrame()
dff = pd.read_csv(path)
print(dff.columns)
print(dff['sentiment'].unique())
print(len(dff))
preds = []
sentences_zz = []
# p=0
for i,row in dff.iterrows():
    print(i)
    # if(i<2000):continue
    # p=p+1
    # print(p)
    # continue
    row_dict = row.to_dict()
    # print()
    sentence = row['text']
    # print(sentence)
    # sentences_zz.append(sentence)
    # sentence_embeddings = get_mean_pooling_emb(sentence)
    # get_nearest_neighbours(sentence_embeddings)
    pred = emotion_candidates_recognition(sentence,1,df,tokenizer,model)
    preds.append(pred)
# print(sentences_zz)
# # preds = []
# sentence_embeddings = get_mean_pooling_emb(sentences_zz)
# for each_emb in sentence_embeddings:
#   pred = get_nearest_neighbours([each_emb])
#   # print(pred)
#   preds.append(pred)

out_dff = dff
print(len(out_dff))
out_dff['predictions'] = preds

out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_goemotions_dual_bert.csv")




import pandas as pd
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/procesd_emotionlines_emo_ids.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/sentences_wc.csv"
from get_nearest_neighbours import get_nearest_neighbours

# path = r"E:\Projects\emo_detector_new\datasets\twitter_dataset.csv"



# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/ISEAR_dataset_cleaned.csv"
# results_df = pd.DataFrame()
# dff = pd.read_csv(path)
# print(dff.columns)
# print(dff['sentiment'].unique())
# print(len(dff))
# preds = []
# sentences_zz = []
# for i,row in dff.iterrows():
#     # print(i)
#     # if(i>1000):continue
#     row_dict = row.to_dict()
#     # print()
#     sentence = row['text']
#     # print(sentence)
#     sentences_zz.append(sentence)
#     # sentence_embeddings = get_mean_pooling_emb(sentence)
#     # get_nearest_neighbours(sentence_embeddings)
#     # pred = emotion_candidates_recognition(sentence,1)
#     # preds.append(pred)
# # print(sentences_zz)
# preds = []
# sentence_embeddings = get_mean_pooling_emb(sentences_zz)
# for each_emb in sentence_embeddings:
#   pred = get_nearest_neighbours([each_emb])
#   # print(pred)
#   preds.append(pred)
#
# out_dff = dff
# out_dff['predictions'] = preds
#
# out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions\twitter_sentiment_prediction_no_neg.csv")


