import pandas as pd
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/procesd_emotionlines_emo_ids.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/sentences_wc.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/twitter_dataset.csv"
from emotion_candidate_recognition import emotion_candidates_recognition

path = r"E:\Projects\emo_detector_new\datasets/ISEAR_dataset_cleaned.csv"
results_df = pd.DataFrame()
dff = pd.read_csv(path)
print(dff.columns)
print(dff['sentiment'].unique())
print(len(dff))
preds = []
sentences_zz = []
for i,row in dff.iterrows():
    print(i)
    # if(i>10):continue
    row_dict = row.to_dict()
    # print()
    sentence = row['text']
    # print(sentence)
    # sentences_zz.append(sentence)
    # sentence_embeddings = get_mean_pooling_emb(sentence)
    # get_nearest_neighbours(sentence_embeddings)
    pred = emotion_candidates_recognition(sentence,1)
    preds.append(pred)
# print(sentences_zz)
# # preds = []
# sentence_embeddings = get_mean_pooling_emb(sentences_zz)
# for each_emb in sentence_embeddings:
#   pred = get_nearest_neighbours([each_emb])
#   # print(pred)
#   preds.append(pred)

out_dff = dff
out_dff['predictions'] = preds

out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_ISEAR_sentiment_new.csv")
