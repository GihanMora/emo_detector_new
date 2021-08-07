# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import pandas as pd

df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_goemotions_new_vocab_refined.csv")
print(df.columns)
print(df['eight_label'].unique())
print(df['fourteen_label'].unique())
print(len(df))
emos_in = ['joy_ecstasy','fear','amazement_surprise' , 'sadness' ,'anger']
df = df[df.eight_label.isin(emos_in)]
print(len(df))