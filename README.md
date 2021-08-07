# SpeechEmotionRecognition

Determine the importance of speech and text features for the task of emotion recognition.

## Brief Summary

1. Generate transcripts from audio files
2. Extract text features from the transcripts generated.
3. Extract audio features like MSF and eGEMAPS from the audio files.
4. Use the features to generate emotion.

Note: Only two emotions, anger and sadness, were considered for the experiments. These emotions represent their corresponding arousal levels. 

## Libraries required
```
tensorflow==2.4.1
numpy==1.19.2
torch==1.9.0
transformers==4.8.2
pandas==1.2.4
librosa==0.8.1
json==2.0.9
sklearn==0.24.2
tqdm==4.61.1
```

## Directory Structure

For more details, follow the README.md files in the corresponding directories. (_comming soon_)


* STFE : Contains some important object definitions that are necessary for all the computations and experiments for this project.
* text_csv : Contains the csv files of clean audio files. These files have been directly from the MELD dataset.
* noise_csv : Contains all transcripts in .csv files for various configurations of audio files.
* text_test : Contains fils that test only on text features for emotion recognition. Helps in choosing appropriate text features.
* audio_test : Contains files that test only the audio features for emotion recognition. Helps in choosing appropriate audio features.
* audio_text : Contains files essential to determine results for various combinations of noises for text and audio to determine the emotions.
* augmentation : Contains files similar to `audio_text` but solely for testing augmented data.
* result : Contains all results and loss graphs from training.
* utils : Contains files that are helper files which help in extraction, saving, processing, of data into some other location.

Data set currently used is MELD, https://affective-meld.github.io




## Citing


S. Zahiri and J. D. Choi. Emotion Detection on TV Show Transcripts with Sequence-based Convolutional Neural Networks. In The AAAI Workshop on Affective Content Analysis, AFFCON'18, 2018.

S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. ACL 2019.
