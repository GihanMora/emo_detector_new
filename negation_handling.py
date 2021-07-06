negations = ['never','not', 'no', 'didn', "didn't", 'didnt', 'didn t', 'doesn t', 'doesnt', "doesn't", "won't", 'won t', 'wont', "aren't", 'aren t', 'arent', 'don t', "don't", 'dont', 'weren t', "weren't", 'werent', "wasn't", 'wasn t', 'wasnt', 'wouldn t', "wouldn't", 'wouldnt', "can't", 'can t', 'couldn t', "couldn't", 'cant', 'cannot', 'couldnt', 'shouldnt', "shouldn't", 'shouldn t', 'neither', 'impossible', 'didn', 'wasn', 'weren', 'aren', 'don', 'doesn', 'couldn', 'shouldn', 'wouldn', 'won', 'nothing']



def map_opposite_emotions(emo_dict):
  opposite_emotions = {'anticipation': 'surprise',
                       'anger': 'joy',
                       'fear': 'trust',
                       'sadness': 'joy',
                       'trust': 'fear',
                       'joy': 'sadness',
                       'surprise': 'anticipation',
                       'disgust': 'joy',
                        'senerity':'distraction',
                        'joy_ecstasy':'sadness',
                        'sad':'joy',
                        'admire':'disgust',
                        'acceptance':'disgust',
                        'amazement_surprise':'anticipation',
                        'distraction':'senerity',
                        'boredom':'interest_vigilance',
                        'disgust_loathing':'joy',
                        'interest_vigilance':'boredom',
                           'negative': 'positive',
                           'positive': 'negative',
                           'model_strong': 'model_weak',
                           'model_weak': 'model_strong',
                       'uncertainty':'uncertainty',
                       'litigious':'litigious'
                       }
  opposed_dict = {}

  for each_key in emo_dict.keys():
    opposite_emo = opposite_emotions[each_key]
    if(opposite_emo in opposed_dict):
      opposed_dict[opposite_emo] = opposed_dict[opposite_emo]+emo_dict[each_key]
    else:
      opposed_dict[opposite_emo] = emo_dict[each_key]

  print(opposed_dict)

  return opposed_dict


# map_opposite_emotions({'anger': 0.353, 'disgust': 0.688,'anticipation':0.009})