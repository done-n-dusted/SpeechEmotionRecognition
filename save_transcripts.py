import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from STFE.SpeechTextFeatures import *
import pandas as pd
import os
from tqdm import tqdm



mname_sp = 'facebook/wav2vec2-base-960h'
mname_txt = 'bert-base-uncased'

SPR = Speech_Recognizer(mname_sp)   

def save_transcripts(source, set_name, class_name, noise_name, db):
    df = pd.DataFrame(columns = ['Utterance', 'Class', 'ID'])

    dir_name = source + set_name + '_' + class_name + '_' + noise_name + '_' + db

    # files = [dir_name + '/' + x for x in os.listdir(dir_name)]
    files = os.listdir(dir_name)
    for i in tqdm(range(len(files))):
        floc = dir_name + '/' + files[i]
        # print(files[i])
        # print(SPR.transcribe(files[i]))
        df.loc[i] = [SPR.transcribe(floc), class_name, files[i]]

    return df



source = '../MELD_noise/'
set_names = ['train', 'test', 'dev']
noises = ['airport', 'babble']
# set_name = 'dev'
# noise_name = 'airport'
dbs = ['0dB', '10dB', '20dB']

for set_name in set_names:
    for noise_name in noises:
        for db in dbs:

            anger_df = save_transcripts(source, set_name, 'anger', noise_name, db)
            sadness_df =  save_transcripts(source, set_name, 'sad', noise_name, db)

            result = anger_df.append(sadness_df, sort = True, ignore_index = True)


            name = './noise_csv/' + set_name + '_' + noise_name + '_' + db + '.csv'
            print(name, 'Done')

            result.to_csv(name)