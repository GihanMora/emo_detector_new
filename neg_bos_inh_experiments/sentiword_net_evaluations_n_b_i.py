import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn
import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix

def compute_metrics(pred, ground_labels):
    labels_all = ground_labels
    preds_all = list(pred)

    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all)
    acc = accuracy_score(labels_all, preds_all)
    confusion_mat = confusion_matrix(labels_all, preds_all)

    out_dict = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusiton_mat': confusion_mat
    }
    for k in out_dict.keys():
        print(k)
        print(out_dict[k])

def senti_word(doc):

    sentences = nltk.sent_tokenize(doc)
    stokens = [nltk.word_tokenize(sent) for sent in sentences]
    taggedlist = []
    for stoken in stokens:
        taggedlist.append(nltk.pos_tag(stoken))
    wnl = nltk.WordNetLemmatizer()

    score_list = []
    for idx, taggedsent in enumerate(taggedlist):
        score_list.append([])
        for idx2, t in enumerate(taggedsent):
            newtag = ''
            lemmatized = wnl.lemmatize(t[0])
            if t[1].startswith('NN'):
                newtag = 'n'
            elif t[1].startswith('JJ'):
                newtag = 'a'
            elif t[1].startswith('V'):
                newtag = 'v'
            elif t[1].startswith('R'):
                newtag = 'r'
            else:
                newtag = ''
            if (newtag != ''):
                synsets = list(swn.senti_synsets(lemmatized, newtag))
                # Getting average of all possible sentiments, as you requested
                score = 0
                if (len(synsets) > 0):
                    for syn in synsets:
                        score += syn.pos_score() - syn.neg_score()
                    score_list[idx].append(score / len(synsets))

    # print(score_list)
    sentence_sentiment = []
    try:
        for score_sent in score_list:
            sentence_sentiment.append(sum([word_score for word_score in score_sent]) / len(score_sent))
        print("Sentiment for each sentence for:" + doc)
        print(sentence_sentiment)
        out_senti = sum(sentence_sentiment)/len(sentence_sentiment)
        print(out_senti)
        return out_senti
    except ZeroDivisionError:
        return 'null'


# senti_word("my name sad khan. may name is good")

# path = r"E:\Projects\emo_detector_new\datasets\twitter_dataset.csv"
# path = r"E:\Projects\emo_detector_new\neg_bos_inh_experiments\fairy_tales_completed_new.csv"
# results_df = pd.DataFrame()
# dff = pd.read_csv(path)
# preds = []
# p_int = []
# preds1 = []
# preds2 = []
# preds3 = []
# for i,row in dff.iterrows():
#     print(i)
#     # if(i>2000):continue
#     row_dict = row.to_dict()
#     # print()
#     # sentence = row['text']
#     text = row['text']
#     text_1 = row['text_negated']  # affective_text
#     text_2 = row['text_boosted']  # affective_text
#     text_3 = row['text_inhibited']  # affective_text
#
#     # print(sentence)
#     # sentences_zz.append(sentence)
#     # sentence_embeddings = get_mean_pooling_emb(sentence)
#     # get_nearest_neighbours(sentence_embeddings)
#     # df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv")
#     # df = df.dropna()
#     # df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]
#     pred = senti_word(text)
#     pred1 = senti_word(text_1)
#     pred2 = senti_word(text_2)
#     pred3 = senti_word(text_3)
#
#     if(pred1=='null'): pred_id_1 = 'ukn'
#     elif(pred1<0):pred_id_1 = 'negative'
#     elif (pred1> 0): pred_id_1 = 'positive'
#     elif(pred1==0):pred_id_1 = 'neutral'
#
#     if (pred == 'null'):
#         pred_id = 'ukn'
#         p_i = abs(0)
#     elif (pred < 0):
#         pred_id = 'negative'
#         p_i = abs(pred)
#     elif (pred > 0):
#         pred_id = 'positive'
#         p_i = abs(pred)
#     elif (pred == 0):
#         pred_id = 'neutral'
#         p_i = abs(pred)
#
#     if (pred2 == 'null'):
#         pred_id_2 = 0
#     elif (pred2 < 0):
#         pred_id_2 = abs(pred2)
#     elif (pred2 > 0):
#         pred_id_2 = abs(pred2)
#     elif (pred2 == 0):
#         pred_id_2 = abs(pred2)
#
#     if (pred3 == 'null'):
#         pred_id_3 = 0
#     elif (pred3 < 0):
#         pred_id_3 = abs(pred3)
#     elif (pred3 > 0):
#         pred_id_3 = abs(pred3)
#     elif (pred3 == 0):
#         pred_id_3 = abs(pred3)
#
#     p_int.append(p_i)
#     preds.append(pred_id)
#     preds1.append(pred_id_1)
#     preds2.append(pred_id_2)
#     preds3.append(pred_id_3)


# out_dff = dff
# out_dff['senti_text'] = preds
# out_dff['senti_text_int'] = p_int
# out_dff['senti_neg'] = preds1
# out_dff['senti_bos'] = preds2
# out_dff['senti_inh'] = preds3
# out_dff.to_csv(r"E:\Projects\emo_detector_new\neg_bos_inh_experiments\fairy_tales_completed_new_with_entiword.csv")

out_dff = pd.read_csv(r"E:\Projects\emo_detector_new\neg_bos_inh_experiments\fairy_tales_completed_new_with_entiword.csv")
#
#
print(out_dff.columns)
acts = []
predis = []
for i, row in out_dff.iterrows():

    # ground_t = row['sentiment']
    # if(ground_t=='neutral'):continue
    predi = row['senti_text']
    # predi = row['senti_neg']

    predis.append(predi)

    ground_t = row['sentiment_id']
    # ground_t = row['senti_neg_id']

    if (ground_t ==4):
        ground_id = 'positive'
    elif (ground_t in [6,2,3]):
        ground_id = 'negative'
        # pred_id = 0

    acts.append(ground_id)




    # #goemotions
    # ground_t = row['sentiment']

compute_metrics(acts,predis)