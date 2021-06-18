from datasets import load_dataset


def ft_to_et(emo_dict):

    eight_dict = {'sadness': 0, 'anger': 0, 'joy': 0, 'anticipation': 0, 'fear': 0, 'surprise': 0, 'disgust': 0,
                  'trust': 0}

    if ('anticipation' in emo_dict.keys()):
        eight_dict['anticipation'] += emo_dict['anticipation']
    if ('amazement_surprise' in emo_dict.keys()):
        eight_dict['surprise'] += emo_dict['amazement_surprise']
    if ('distraction' in emo_dict.keys()):
        eight_dict['surprise'] += emo_dict['distraction']
    if ('fear' in emo_dict.keys()):
        eight_dict['fear'] += emo_dict['fear']
    if ('trust' in emo_dict.keys()):
        eight_dict['trust'] += emo_dict['trust']
    if ('interest_vigilance' in emo_dict.keys()):
        eight_dict['anticipation'] += emo_dict['interest_vigilance']
    if ('anger' in emo_dict.keys()):
        eight_dict['anger'] += emo_dict['anger']
    if ('disgust_loathing' in emo_dict.keys()):
        eight_dict['disgust'] += emo_dict['disgust_loathing']
    if ('senerity' in emo_dict.keys()):
        eight_dict['joy'] += emo_dict['senerity']
    if ('boredom' in emo_dict.keys()):
        eight_dict['disgust'] += emo_dict['boredom']
    if ('acceptance' in emo_dict.keys()):
        eight_dict['trust'] += emo_dict['acceptance']
    if ('joy_ecstasy' in emo_dict.keys()):
        eight_dict['joy'] += emo_dict['joy_ecstasy']
    if ('admire' in emo_dict.keys()):
        eight_dict['trust'] += emo_dict['admire']
    if ('sadness' in emo_dict.keys()):
        eight_dict['sadness'] += emo_dict['sadness']

    print(eight_dict)
    return eight_dict


# emo_dict = {'senerity': 0.011, 'amazement_surprise': 0.015, 'admire': 0.016, 'trust': 0.033, 'distraction': 0.041, 'anger': 0.043, 'disgust_loathing': 0.06, 'anticipation': 0.071, 'sadness': 0.084, 'joy_ecstasy': 0.087, 'fear': 0.105, 'boredom': 0.169, 'acceptance': 0.306}
#
# ft_to_et(emo_dict)

import pandas as pd

df = pd.read_csv(r'E:\Data\Emotion_detection_gihan\goemotions\data\test.csv')
emo_go_emotions = {
1:'admiration',
2:'amusement',
3:'anger',
4:'annoyance',
# 5:'approval',
5:'trust',
6:'caring',
7:'confusion',
8:'curiosity',
9:'desire',
10:'disappointment',
11:'disapproval',
12:'disgust',
13:'embarrassment',
14:'excitement',
15:'fear',
16:'gratitude',
17:'grief',
18:'joy',
19:'love',
20:'nervousness',
21:'optimism',
22:'pride',
23:'realization',
24:'relief',
25:'remorse',
26:'sadness',
27:'surprise',
28:'neutral'}

print('sad',len(df.loc[df['label'] == 26]))
print('grief',len(df.loc[df['label'] == 17]))
print('dissapointment',len(df.loc[df['label'] == 10]))

print('joy',len(df.loc[df['label'] == 18]))
print('excitement',len(df.loc[df['label'] == 14]))

print('anger',len(df.loc[df['label'] == 3]))
print('annoy',len(df.loc[df['label'] == 4]))

print('surprise',len(df.loc[df['label'] == 27]))


print('disgust',len(df.loc[df['label'] == 12]))

print('trust',len(df.loc[df['label'] == 5]))

print('fear',len(df.loc[df['label'] == 15]))
print('nervousness',len(df.loc[df['label'] == 20]))
# for i,row in df.iterrows():
#     if(row['label']==26):
#         print(i,row)
print(len(df))
df_s = df.loc[df['label'] == 27][:400]
df = df.loc[df['label'].isin([26,17,10,18,14,3,4,27,12,5,15,20])]
print('interested',len(df))
df = df.loc[df['label'] != 27]

print('no sup',len(df))

df = pd.concat([df,df_s,], ignore_index=True)
print(len(df))
new_labels = []

for i,row in df.iterrows():
    if row['label'] in [26,17,10]:
        new_labels.append(0)
    elif row['label'] in [14,18]:
        new_labels.append(1)
    elif row['label'] in [3,4]:
        new_labels.append(2)
    elif row['label'] in [27]:
        new_labels.append(3)
    elif row['label'] in [12]:
        new_labels.append(4)
    elif row['label'] in [5]:
        new_labels.append(5)
    elif row['label'] in [15,20]:
        new_labels.append(6)
print(len(df))

df['label'] = new_labels
df.to_csv(r'E:\Data\Emotion_detection_gihan\goemotions\data\test_simple.csv')


emotions = load_dataset('csv', data_files={'train': [r'E:\Data\Emotion_detection_gihan\goemotions\data\train.csv'],
                                          'validation': [r'E:\Data\Emotion_detection_gihan\goemotions\data\test.csv']})

