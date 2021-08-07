import sys

sys.path.insert(1, '../')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from STFE.SpeechTextFeatures import *
from tqdm import tqdm
import re

mname_txt = 'bert-base-uncased'
mname_asr = 'facebook/wav2vec2-base-960h'

STTFE = Speech_To_Text_Features(mname_asr, mname_txt)

types = 'ABCDEFGHIJKLMNPQR'

from_folder = '../../../mitacs/MELD_clean_aug/'
to_folder = '../../../mitacs/MELD_clean_aug_text/'

sets = ['dev', 'train']
emos = ['anger', 'sad']
original = []

for s in sets:
    for emo in emos:
        original.append(s + '_' + emo + '_clean')
        for c in types:
            original.append(c + '_' + s + '_' + emo + '_clean')


# print(original)
def get_file_name(folder, original):
    res = []

    for f in original:
        res += [f  + '/' + x for x in os.listdir(folder + f)]

    return res

all_audio_files = get_file_name(from_folder, original)
# print(all_audio_files)
for one_file in tqdm(all_audio_files):
    try:
        feats = pd.DataFrame(STTFE.get_text_features(from_folder + one_file))
        feats.to_csv(to_folder + one_file[:-3] + 'csv', index = False)
    except Exception as e:
        print(to_folder + one_file)
        print(e)
    
    # break
