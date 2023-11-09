# Preparing MEAD video data
In order to train EMOTE, you will need the processed MEAD dataset. 

1) Go to the [MEAD website](https://github.com/uniBruce/Mead) and download the dataset.

2) Download the processed data by running:

```bash
download_processed_mead.sh
```
Instead of downloading the processed data, you can also process the dataset yourself. You can find the instructions on that below.

3) Coming soon


# Processing MEAD video data processing 

This README will describe all you need to process the MEAD dataset. 

This includes the following steps: 
1) Face and landmark detection (mediapipe and FAN)
2) Cropped video extraction 
3) Audio extraction 
4) Segmentation 
5) Emotion feature extraction 
6) Pseudo-GT 3D face reconstruction 

Detailed step-by-step manual is coming soon. Meanwhile you can inspect the [`process_mead.py`](./process_mead.py) script.

