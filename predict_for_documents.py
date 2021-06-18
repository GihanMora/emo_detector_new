import nltk
import pandas as pd
nltk.download('punkt')
from emotion_candidate_recognition import emotion_candidates_recognition

text = "Sandali is a beautiful girl but she is annoying at times. But i like her personality and attitudes."
a_list = nltk.tokenize.sent_tokenize(text)
for each_s in a_list:
    pred = emotion_candidates_recognition(each_s,1)
    print(pred)

# print(pred)