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
# path = r"E:\Projects\emo_detector_new\datasets/ISEAR_dataset_cleaned.csv"
path= r"E:\Projects\Emotion_detection_gihan\finbert_experiments\financial phrasebank\processed_fpbank.csv"
results_df = pd.DataFrame()
dff = pd.read_csv(path)

preds = []
for i,row in dff.iterrows():
    print(i)
    # if(i>2000):continue
    row_dict = row.to_dict()
    # print()
    # sentence = row['text']
    sentence = row['sentence']
    # print(sentence)
    # sentences_zz.append(sentence)
    # sentence_embeddings = get_mean_pooling_emb(sentence)
    # get_nearest_neighbours(sentence_embeddings)
    # df = pd.read_csv(r"E:\Projects\emo_detector_new\vocabs\mean_pooling_emb_emobert_new_vocab_refined.csv")
    # df = df.dropna()
    # df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]
    pred = senti_word(sentence)
    if(pred=='null'): pred_id = 'ukn'
    elif(pred<0):pred_id = 'negative'
    elif (pred > 0): pred_id = 'positive'
    elif(pred==0):pred_id = 'neutral'

    preds.append(pred_id)
# print(sentences_zz)
# # preds = []
# sentence_embeddings = get_mean_pooling_emb(sentences_zz)
# for each_emb in sentence_embeddings:
#   pred = get_nearest_neighbours([each_emb])
#   # print(pred)
#   preds.append(pred)

out_dff = dff
out_dff['predictions'] = preds
out_dff.to_csv(r"E:\Projects\emo_detector_new\predictions/prediction_financialpb_sentiword_net.csv")




acts = []
predis = []
for i, row in out_dff.iterrows():

    # ground_t = row['sentiment']
    # if(ground_t=='neutral'):continue


    ground_t = row['sentiment_id']

    if (ground_t ==1):
        ground_id = 'positive'
    elif (ground_t in [2,3,4,5]):
        ground_id = 'negative'
        # pred_id = 0
    else:
        continue

    acts.append(ground_id)

    predi = row['predictions']

    predis.append(predi)




    # #goemotions
    # ground_t = row['sentiment']

compute_metrics(acts,predis)