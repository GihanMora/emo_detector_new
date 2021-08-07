negative = [
'down from'
,'lower growth'
,'lowered profit'
,'reduced profit'
,'below the level'
,'profit warning'
,'profilt is lower'
,'profit decrease'
,'less sales'
,'less effective'
,'down from EUR'
,'profit slightly down'
,'sales slightly down'
,'down to EUR'
,'down EUR'
,'downed to EUR'
]
positive = [
'fasten'
,'great'
,'improve'
,'increase'
,'up'
,'upper'
,'totalled'
,'Cost cutting'
,'dampened loss'
,'optimization'
,'reduce loss'
,'increase profit'
,'enhance profit'
,'cut cost'
,'reduce lost'
,'diluted EPS'
,'diluted loss'
,'diluted lost'
,'totalled compared to loss'
,'increased compared to loss'
,'doubled compared to loss'
,'upgraded from underperformed'
,'climbed 1.21 %'
,'up from a loss'
,'up from EUR'
,'rose to EUR'
,'increased to EUR'
,'smaller lost'
,'costs fell'
,'loss narrowed'
,'grew by 40 %'
,'increase by 40 %']
uncertainty = []
litigious = []
model_strong = []
model_weak = []

negative= [i.lower() for i in negative]
positive= [i.lower() for i in positive]
uncertainty= [i.lower() for i in uncertainty]
litigious= [i.lower() for i in litigious]
model_strong= [i.lower() for i in model_strong]
model_weak= [i.lower() for i in model_weak]

lnm_dict = {'negative':negative,'positive':positive,'uncertainty':uncertainty,'litigious':litigious,'model_strong':model_strong,'model_weak':model_weak}
print(lnm_dict)

a = ['a','b','c']
b= ['a','b']
print(list(set(a)-set(b)))

# from google.colab import drive
# drive.mount('/content/drive')
# !pip install transformers
import numpy
import pandas as pd
import ast
from datetime import datetime
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModel, BertModel,AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import re               # Obtain expressions
from gensim.models import Word2Vec    #Import gensim Word2Fec
from sklearn.decomposition import PCA #Grab PCA functions
import numpy as np
#Plot helpers
import matplotlib
import matplotlib.pyplot as plt
#Enable matplotlib to be interactive (zoom etc)
import ast
import pandas as pd
import ast
from datetime import datetime
# df = pd.read_csv(r"/content/drive/MyDrive/NLP_notebooks/Emotional_embeddings/mean_pooling_embeddings_emo_bert.csv")
# print(df.head())
# df = df.dropna()
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
dis = cosine_similarity([[1, 0, -1]], [[1,-1, 0]])
# print(dis)


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


out_df = pd.DataFrame()
out_df['token'] = tokens
out_df['embedding'] = embedding
out_df['eight_label'] = eight_label
out_df['fourteen_label'] = fourteen_label

existing_def = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_even_pn.csv")

df = pd.concat([existing_def, out_df])

df.to_csv(r"E:\Projects\emo_detector_new\vocabs\lnm_vocab_even_pn_extended_2.csv")