import pandas as pd
import numpy as np
from sklearn import preprocessing
from STFE.SpeechTextFeatures import *
from tqdm import tqdm
import re

mname_txt = 'bert-base-uncased'
TFE = BERT_Text_Feature_Extracter(mname_txt)

# transcript_path = 'text_csv/dev_sent_emo.csv'

transcript_path = 'noise_csv/'
set = 'test'
noise = 'CAFETERIA'
target = '../../mitacs/cafeteria_text/'


if noise == 'clean':
    data_frame = pd.read_csv(transcript_path)
    req = np.array(data_frame[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']])
    print('\nPreparing data for ' + set)
    for u, e, d, uid in tqdm(req):
        # print(d)
        if e == 'anger' or e == 'sadness':

            u = re.sub(r'[^\w\s]','',u)    
            u = u.lower()
            feats = pd.DataFrame(TFE.features_fromtext(u))
            if e == 'sadness': e = 'sad'
            feats.to_csv(target + set + '_' + e + '_' + noise + '/dia' + str(d) + '_utt' + str(uid) + '.csv', index = False)

else:

    # for s in ['train', 'test', 'dev']:
    for s in ['test']:
        # for n in ['airport_0dB', 'airport_10dB', 'airport_20dB', 'babble_0dB', 'babble_10dB', 'babble_20dB']:
        for n in ['CAFETERIA_0dB', 'CAFETERIA_5dB', 'CAFETERIA_10dB', 'CAFETERIA_15dB', 'CAFETERIA_20dB']:
            data_frame = pd.read_csv(transcript_path + s + '_' + n + '.csv')

            req = np.array(data_frame[['Utterance', 'Class', 'ID']])
            print('\nPreparing data for ' + s + '_' + n)

            for u, e, id in tqdm(req):
                
                if pd.isna(u) == False:
                    feats = pd.DataFrame(TFE.features_fromtext(u))
                    id = id[:-3] + 'csv'
                    feats.to_csv(target + s + '_' + e + '_' + n + '/' + id, index = False)
                    
                # print(id, target + s + '_' + e + '_' + n + '/' + id)
                # break






