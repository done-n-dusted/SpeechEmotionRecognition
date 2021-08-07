# `audio_test` Directory

This directory has files that help in choosing the right audio features or a combination of those features that would help in the task of emotion recognition.

The chosen audio features, or the combination of those features are combined with the appropriate text features of the speech for the final task.

The various features that have been tested out are:
* Prosody
* MRS
* MSF
* eGEMAPS

Each file helps in evaluating the results for the task for every audio feature. The following files represent contain code for the process of evaluation.

| File Name | Feature being Evaluated |
|-----------|-------------------------:|
|`aud_pro.py`|Prosody|
|`aud_mrs.py`|MRS|
|`aud_msf.py`|MSF|
|`aud_egemaps.py`|eGEMAPS|
|`aud_mrs_pro.py`|MRS and Prosody|
|`aud_msf_pro.py`|MSF and Prosody|
|`aud_pro_gmap.py`|Prosody and eGEMAPS|
|`aud_msf_egempas.py`| MSF and eGEMPAS|

From the results achieved, it was concluded that a combination of MSF features and eGEMAPS features worked well for the task of emotion recognition.
