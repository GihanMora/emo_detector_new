import pandas as pd

df = pd.read_csv(r"E:\Projects\emo_detector_new\datasets\sem_eval_2018.csv")
texts = []
labels1 = []
labels2 = []

for i,row in df.iterrows():
    l = []

    if(row['anticipation']==1):
        l.append('anticipation')
    if (row['disgust'] == 1):
        l.append('disgust')
    if (row['surprise'] == 1):
        l.append('surprise')
    if (row['trust'] == 1):
        l.append('trust')




    if(len(l)==1):
        labels1.append(l[0])
        labels2.append(l[0])
        texts.append(row['Tweet'])
    if (len(l) == 2):
        labels1.append(l[0])
        labels2.append(l[1])
        texts.append(row['Tweet'])



df_out = pd.DataFrame()
df_out['text'] = texts
df_out['labels1'] = labels1
df_out['labels2'] = labels2


df_out.to_csv("E:\Projects\emo_detector_new\datasets\semeval_18_processed_final.csv")


