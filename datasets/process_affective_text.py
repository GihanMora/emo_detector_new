from xml.dom import minidom
import numpy as np
xmldoc = minidom.parse(r'E:\Projects\emo_detector_new\datasets\affectivetext_trial.xml')
itemlist = xmldoc.getElementsByTagName('instance')
texts = []
for k in itemlist:
    print(k.firstChild.nodeValue)
    texts.append(k.firstChild.nodeValue)

f = open(r"E:\Projects\emo_detector_new\datasets\affectivetext_trial.emotions.txt",'r')
sent_ids = []
sents = []
for each_l  in f.readlines():
    emo_dict = {0:'anger', 1:'disgust',2: 'fear',3: 'joy',4: 'sadness',5: 'surprise'}
    vals = [int(i) for i in each_l.strip().split(' ')[1:]]
    print(np.argmax(vals))
    print(vals)
    sent_ids.append(np.argmax(vals))
    sents.append(emo_dict[np.argmax(vals)])



import pandas as pd

df = pd.DataFrame()
df['text'] = texts
df['sentiment'] = sents
df['sentiment_id'] = sent_ids


# df.to_csv(r"E:\Projects\emo_detector_new\datasets\affectivetext_trial.csv")
df1 = pd.read_csv(r"E:\Projects\emo_detector_new\datasets\affectivetext_trial.csv")
df2 = pd.read_csv(r"E:\Projects\emo_detector_new\datasets\affectivetext_test.csv")
dddf = pd.concat([df1,df2], ignore_index=True)
print(len(dddf))
# print(dddf['predictions'].unique())

dddf.to_csv(r"E:\Projects\emo_detector_new\datasets\affectivetext_full.csv")