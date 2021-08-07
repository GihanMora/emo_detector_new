import ast

import pandas as pd
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/procesd_emotionlines_emo_ids.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/sentences_wc.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/twitter_dataset.csv"
from transformers import AutoTokenizer, AutoModel

from emotion_candidate_recognition import emotion_candidates_recognition

# path = r"E:\Projects\emo_detector_new\datasets\twitter_dataset.csv"
# path = r"E:\Projects\emo_detector_new\datasets/ISEAR_dataset_cleaned.csv"
# path = r"E:\Projects\emo_detector_new\datasets\fairy_tales_full.csv"
# path = r"E:\Projects\emo_detector_new\datasets\affectivetext_full.csv"
# path = r"E:\Projects\emo_detector_new\datasets\semeval_18_processed_final.csv"
# path = r"E:\Projects\emo_detector_new\datasets\test_senti4sd.csv"
path= r"E:\Projects\Emotion_detection_gihan\finbert_experiments\financial phrasebank\processed_fpbank.csv"
results_df = pd.DataFrame()
dff = pd.read_csv(path)
print(dff.columns)
# print(dff['sentiment'].unique())
print(len(dff))

import ast
tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\go_model")
model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\go_model")


# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv")
# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_goemotions_new_vocab_refined.csv")
# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_goemotions_new_vocab.csv")
# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_goemo_p_n_j_s.csv")
df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_goemo_p_n_j_s_extended.csv")
# emos_in = ['trust' ,'amazement_surprise','disgust_loathing','anticipation']
# df = df[df.fourteen_label.isin(emos_in)]
df = df.dropna()
df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]



preds = []
sentences_zz = []
for i,row in dff.iterrows():
    print(i)
    # if(i>2000):continue
    row_dict = row.to_dict()
    # print()
    sentence = row['sentence']
    # sentence = row['text']
    # print(sentence)
    # sentences_zz.append(sentence)
    # sentence_embeddings = get_mean_pooling_emb(sentence)
    # get_nearest_neighbours(sentence_embeddings)
    # df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv")
    # df = df.dropna()
    # df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]
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
out_dff['predictions'] = preds
# out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_ISEAR_go_emo_model_cuz.csv")
# out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_fairy_tales_go_emo_model_cuz.csv")
# out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_senti4SD_go_emo_mode.csv")
out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_fpb_go_emo_lnm_extended.csv")
# out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_twitter.csv")
