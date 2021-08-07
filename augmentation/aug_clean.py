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
to_folder = '../../../mitacs/MELD_clean_aug/'

sets = ['dev', 'train']
emos = ['anger', 'sad']
original = []

for s in sets:
    for emo in emos:
        original.append(s + '_' + emo + '_clean')

def get_file_name(folder, original):
    res = []

    for f in original:
        res += [f  + '/' + x for x in os.listdir(folder + f)]

    return res

#has all the file_list
all_audio_files = get_file_name(from_folder, original)

for one_file in tqdm(all_audio_files):

    # try:
        input_audio, _ = librosa.load(from_folder + one_file, sr = reqrate, res_type = 'kaiser_fast')
        # print(input_audio)
        sf.write(to_folder + one_file, input_audio, reqrate)

        aug_file, _ = audaugs.add_background_noise(input_audio)
        sf.write(to_folder + 'A_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.apply_lambda(input_audio)
        sf.write(to_folder + 'B_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.change_volume(input_audio)
        sf.write(to_folder + 'C_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.clicks(input_audio)
        sf.write(to_folder + 'D_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.clip(input_audio)
        sf.write(to_folder + 'E_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.harmonic(input_audio)
        sf.write(to_folder + 'F_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.high_pass_filter(input_audio)
        sf.write(to_folder + 'G_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.insert_in_background(input_audio)
        sf.write(to_folder + 'H_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.invert_channels(input_audio)
        sf.write(to_folder + 'I_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.low_pass_filter(input_audio)
        sf.write(to_folder + 'J_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.normalize(input_audio)
        sf.write(to_folder + 'K_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.peaking_equalizer(input_audio)
        sf.write(to_folder + 'L_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.percussive(input_audio)
        sf.write(to_folder + 'M_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.pitch_shift(input_audio)
        sf.write(to_folder + 'N_' + one_file, aug_file, reqrate)
        
        # aug_file, _ = audaugs.reverb(input_audio)
        # sf.write(to_folder + 'O_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.speed(input_audio)
        sf.write(to_folder + 'P_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.tempo(input_audio)
        sf.write(to_folder + 'Q_' + one_file, aug_file, reqrate)
        
        aug_file, _ = audaugs.time_stretch(input_audio)
        sf.write(to_folder + 'R_' + one_file, aug_file, reqrate)

    # except RuntimeError:
        # print('Caught RuntimeError at', to_folder + one_file)

# print(all_audio_files)

'''
MAPPING

  A  = add_background_noise
  B  = apply_lambda
  C  = change_volume
  D  = clicks
  E  = clip
  F  = harmonic
  G = high_pass_filter
  H  = insert_in_background
  I  = invert_channels
  J  = low_pass_filter
  K  = normalize
  L  = peaking_equalizer
  M  = percussive
  N  = pitch_shift
  O  = reverb # not working for some reason
  P  = speed
  Q  = tempo
  R  = time_stretch
  S  = to_mono

'''