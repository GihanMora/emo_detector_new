import pandas as pd


# from nltk.stem import PorterStemmer
# porter = PorterStemmer()
# df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab.csv")
# df = df.loc[df['eight_label'] == 'model_strong']
# print(df.head())
# print(df.columns)
# print(len(df))
# stemmed_tokens = []
# for i,row in df.iterrows():
#     # print(row['token'])
#     stemmed_tokens.append(porter.stem(row['token']))
#
# df['stemmed_token'] = stemmed_tokens
#
# unique_stems = df['stemmed_token'].unique()
#
# print(len(unique_stems))
# #
# #
# f_tokens = []
# f_embeddings = []
# f_e_l = []
# f_ft_l = []
# #
# #
# for stem in unique_stems:
#     # print(stem)
#     for i,row in df.iterrows():
#         if(row['stemmed_token']==stem):
#             print(row['token'])
#             f_tokens.append(row['token'])
#             f_embeddings.append(row['embedding'])
#             f_e_l.append(row['eight_label'])
#             f_ft_l.append(row['fourteen_label'])
#             break
# #     if(row['eight_label']=='positive'):
# #         unique_stems.append(row['stemmed_token'])
#
#
# dff = pd.DataFrame()
# dff['token'] = f_tokens
# dff['embedding'] = f_embeddings
# dff['eight_label'] = f_e_l
# dff['fourteen_label'] = f_ft_l
#
# dff = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab.csv")
# dff = df.loc[df['eight_label'] == 'model_strong']
# print(len(dff))
# dff.to_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_ms.csv")



# df1 = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_l.csv")
# df2 = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_ms.csv")
# df3 = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_mw.csv")
# df4 = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_n.csv")
# df5 = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_p.csv")
# df6 = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_u.csv")
# dddf = pd.concat([df1,df2,df3,df4,df5,df6], ignore_index=True)
# print(len(dddf))
# dddf.to_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_final.csv")
df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_final.csv")
print(len(df))
print(len(df.loc[df['eight_label'] == 'negative']))
for i,row in df.iterrows():
    if((i%2==0 or i%3==0 or i%5==0) and (row['eight_label'] == 'negative')):
        df.drop(i, inplace=True)
    if ((i % 2 == 0) and (row['eight_label'] == 'litigious')):
        df.drop(i, inplace=True)

print(len(df.loc[df['eight_label'] == 'positive']))
print(len(df.loc[df['eight_label'] == 'negative']))
print(len(df.loc[df['eight_label'] == 'uncertainty']))
print(len(df.loc[df['eight_label'] == 'litigious']))
print(len(df.loc[df['eight_label'] == 'model_strong']))
print(len(df.loc[df['eight_label'] == 'model_weak']))

df.to_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_even.csv")