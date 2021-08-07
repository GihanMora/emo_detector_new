import pickle
import torch
from transformers import AutoTokenizer, AutoModel

with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/emotions/emotions_plutchik.pkl', 'rb') as f:
    EMOTION_MAP = pickle.load(f)

lnm_dict = EMOTION_MAP
print(lnm_dict)




#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\emo_bert_model")
model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\emo_bert_model")

# tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
# model = AutoModel.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")

tokens = []
embedding = []
eight_label = []
fourteen_label = []

for each_key in lnm_dict:
  words_list = lnm_dict[each_key]

  for wd in words_list:
      print(wd)
      sentences = [wd]
      encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
      with torch.no_grad():
          model_output = model(**encoded_input)
      sentence_embeddings_raw = mean_pooling(model_output, encoded_input['attention_mask'])
      sentence_embeddings = sentence_embeddings_raw.tolist()[0]
      tk = wd
      emb = sentence_embeddings
      e_lbl = each_key

      tokens.append(tk)
      embedding.append(emb)
      eight_label.append(e_lbl)
      fourteen_label.append(e_lbl)

    # print(tk)
    # print(emb)
    # print(e_lbl)
    # print(ft_lbl)
  #  break
import pandas as pd

out_df = pd.DataFrame()
out_df['token'] = tokens
out_df['embedding'] = embedding
out_df['eight_label'] = eight_label
out_df['fourteen_label'] = eight_label


out_df.to_csv(r"E:\Projects\emo_detector_new\vocabs\plutchik_with_emobert_vocab.csv")