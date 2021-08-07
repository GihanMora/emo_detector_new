import pandas as pd
from transformers import AutoTokenizer, AutoModel
from emotion_candidate_recognition import emotion_candidates_recognition
import ast


model_path = r"E:\Projects\emo_detector_new\emo_bert_model"
vocab_path = r"E:\Projects\emo_detector_new\vocabs\plutchik_with_emobert_vocab.csv"



tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)


df = pd.read_csv(vocab_path)
df = df.dropna()
df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]

sentence = "my name is khan im not a terrorist"
pred = emotion_candidates_recognition(sentence,1,df,tokenizer,model)

print(pred)