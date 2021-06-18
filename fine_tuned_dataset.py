from datasets import load_dataset
# emotions = load_dataset("emotion")
# emotions = load_dataset("go_emotions", "raw")
emotions = load_dataset('csv', data_files={'train': [r'E:\Data\Emotion_detection_gihan\goemotions\data\train.csv'],
                                          'validation': [r'E:\Data\Emotion_detection_gihan\goemotions\data\test.csv']})
#https://huggingface.co/datasets/emotion
print(emotions['train']['label'])

# print(type(emotions))


# import pandas as pd
#
# train_df = pd.read_csv(r'E:\Data\Emotion_detection_gihan\goemotions\data\test.csv')
#
# labels = []
#
# for i,row in train_df.iterrows():
#     labels.append(int(row['label'].split(',')[0]))
#
#
#
# train_df['label'] = labels
#
# train_df.to_csv(r'E:\Data\Emotion_detection_gihan\goemotions\data\test.csv')