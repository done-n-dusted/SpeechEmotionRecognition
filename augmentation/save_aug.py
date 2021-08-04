# Saves augmented audio files in the tree

import sys

sys.path.insert(1, '../')

import os
import augly.audio as audaugs
import augly.utils as utils
import librosa
import glob 
from tqdm import tqdm
import soundfile as sf

reqrate = 16000

from_folder = '../../../mitacs/MELD_noise/'
to_folder = '../../../mitacs/MELD_noise_aug/'

# l = glob.glob(from_folder + '*/*')

def get2levels(folder):
    res = []
    par_dirs = os.listdir(folder)
    # print(par_dirs)
    for f in par_dirs:
        # print(f)
        res += [f + '/' + x for x in os.listdir(folder + f)]
    return res


# l = list(os.walk(from_folder))
all_audio_files = get2levels(from_folder)
# print(all_audio_files[0], len(all_audio_files))
for one_file in tqdm(all_audio_files):
    # print(one_file)
    try:
        input_audio, _ = librosa.load(from_folder + one_file, sr = reqrate, res_type = 'kaiser_fast')
        aug_file, _ = audaugs.pitch_shift(input_audio, n_steps = 2.0)
        # print(to_folder + one_file)
        sf.write(to_folder + one_file, aug_file, reqrate)
    except RuntimeError:
        print('Caught RuntimeError at', to_folder + one_file)


