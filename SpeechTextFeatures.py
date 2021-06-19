from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer
from transformers import BertTokenizer, TFBertModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import torch

class Speech_Recognizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForMaskedLM.from_pretrained(self.model_name)

    def transcribe(self, audio_file_name):
        audio_input, sampling_rate = librosa.load(audio_file_name, sr = 160000, res_type = 'kaiser_fast')

        input_values = self.tokenizer(audio_input, return_tensors = 'tf').input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim = -1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
    

class Text_Feature_Extracter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = TFBertModel.from_pretrained(self.model)

    def features_fromtext(self, text_array, window = 50):
        data = np.array([[0]*768])

        for i in tqdm(range(0, len(text_array), window)):
            encoded = self.tokenizer(text_array[i:i+window], padding = 'max_length', return_tensors = 'tf')
            output = model(encoded['input_ids'], attention_mask = encoded['attention_mask'])
            temp = output.pooler_output
            data = np.concatenate([data, temp])
        
        data = data[1:]
        return data

