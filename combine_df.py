import pandas as pd


df1 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_fairy_tales_new_0_600_emobert_14_refined.csv")
df2 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_fairy_tales_new_600_1206_emobert_14_refined.csv")

# df4 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_new_0_1000_emobert_14_refined.csv")
# df5 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_new_1000_2000_emobert_14_refined.csv")
# df6 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_new_2000_3000_emobert_14_refined.csv")
# df7 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_new_3000_4000_emobert_14_refined.csv")
# df8 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_new_4000_5476_emobert_14_refined.csv")
# df9 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_new_4000_5000_gogogo_14.csv")
# df10 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_new_5000_5477_gogogo_14.csv")

# df2 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_fairy_tales_new_400_800_14.csv")
# df3 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_fairy_tales_new_800_1206_14.csv")
# df4 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_affective_text_new_900_1249_14.csv")
# df5 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_sentiment_new_4000_5000_no_bordom.csv")
# df6 = pd.read_csv(r"E:\Projects\emo_detector_new\predictions\predictions_ISEAR_sentiment_new_5000_5476_no_bordom.csv")
# print(len(df1))
# print(len(df2))

dddf = pd.concat([df1,df2], ignore_index=True)
print(len(dddf))
# print(dddf['predictions'].unique())

dddf.to_csv(r"E:\Projects\emo_detector_new\predictions\predictions_fairy_tales_embert_all_14_refined.csv")