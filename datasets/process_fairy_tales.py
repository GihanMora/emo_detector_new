f = open(r"E:\Projects\emo_detector_new\datasets\GrimmsAll4labsagree.txt","r")
texts = []
sentis = []
emos = []
emo_dict = {2:'Angry-Disgusted',3:'Fearful',4:'Happy',6:'Sad',7:'Surprised' }
for each_ft in f.readlines():
    try:
        vals = each_ft.strip().split('@')
        text = vals[2]
        senti = vals[1]
        print(text,senti)
        texts.append(text)
        sentis.append(senti)
        emos.append(emo_dict[int(senti)])
    except: continue


import pandas as pd

df = pd.DataFrame()
df['text'] = texts
df['sentiment_id'] = sentis
df['sentiment'] = emos


# df.to_csv(r"E:\Projects\emo_detector_new\datasets\grimms_processed.csv")

df1 = pd.read_csv(r"E:\Projects\emo_detector_new\datasets\potter_processed.csv")
df2 = pd.read_csv(r"E:\Projects\emo_detector_new\datasets\hcand_processed.csv")
df3 = pd.read_csv(r"E:\Projects\emo_detector_new\datasets\grimms_processed.csv")
dddf = pd.concat([df1,df2,df3], ignore_index=True)
print(len(dddf))
# print(dddf['predictions'].unique())
print(len(dddf))
dddf.to_csv(r"E:\Projects\emo_detector_new\datasets\fairy_tales_full.csv")