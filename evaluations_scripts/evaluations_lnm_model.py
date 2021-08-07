#
import ast

import pandas as pd
from transformers import AutoTokenizer, AutoModel

from emotion_candidate_recognition import emotion_candidates_recognition, emotion_candidates_recognition_lnm

# path = r"E:\Projects\emo_detector_new\datasets\twitter_dataset.csv"
path= r"E:\Projects\Emotion_detection_gihan\finbert_experiments\financial phrasebank\processed_fpbank.csv"

results_df = pd.DataFrame()
dff = pd.read_csv(path)
print(dff.columns)
print(dff['sentiment'].unique())
print(len(dff))
preds = []
sentences_zz = []
# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_final.csv")

# print(len(df1))
# print(df1['eight_label'].unique())

# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_final.csv")
# print(len(df))
# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_even_pn.csv")
# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_even_pn_extended.csv")
df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_even_pn_extended_10down_5_narr.csv")
emos_in = ['negative' ,'positive']
df = df[df.eight_label.isin(emos_in)]
df1 = df.dropna()
print(len(df1))


# df2 = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_goemotions_new_vocab_refined.csv")
# # # df2 = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv")
# emos_in_a = ['sadness' ,'joy_ecstasy']
# df2 = df2[df2.fourteen_label.isin(emos_in_a)]
# df2 = df2.dropna()
# print(len(df2))
#
#
# df =  pd.concat([df, df2])
#
#
# df.to_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_goemo_p_n_j_sad.csv")
print('aa',df['fourteen_label'].unique())

df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]

tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\emo_bert_model")
model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\emo_bert_model")
# tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\go_model")
# model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\go_model")
for i,row in dff.iterrows():
    print(i)

    row_dict = row.to_dict()

    sentence = row['sentence']



    pred = emotion_candidates_recognition_lnm(sentence, 1, df, tokenizer, model)
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
out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_lnm_model_fpb_lnm_goemo_equal_pn_extended_dd_10_down.csv")
# out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_twitter.csv")
