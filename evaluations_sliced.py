import pandas as pd
from emotion_candidate_recognition import emotion_candidates_recognition
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/procesd_emotionlines_emo_ids.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/sentences_wc.csv"
# path = r"/content/drive/MyDrive/NLP_notebooks/emotionlines_dataset/twitter_dataset.csv"


# path = r"E:\Projects\emo_detector_new\datasets\twitter_dataset.csv"
# path = r"E:\Projects\emo_detector_new\datasets/ISEAR_dataset_cleaned.csv"
# path = r"E:\Projects\emo_detector_new\datasets\fairy_tales_full.csv"
path = r"E:\Projects\emo_detector_new\datasets\goemotions.csv"
results_df = pd.DataFrame()
dff = pd.read_csv(path)
# print(dff.columns)
# print(dff['sentiment'].unique())
print(len(dff))
preds = []
sentences_zz = []

def slice_run(start,end):
    for i,row in dff.iterrows():
        print(i)
        if((i>end) or (i<start)):continue
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

    out_dff = dff[start:end+1]
    out_dff['predictions'] = preds
    out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_goemotion_new_"+str(start)+"_"+str(end)+"_emobert_14_refined.csv")
# out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/predictions_twitter.csv")

s_e = input('start and end: ')
s_e = s_e.split('_')

slice_run(int(s_e[0]),int(s_e[1]))

