# Preparing MEAD video data
In order to train EMOTE, you will need the processed MEAD dataset. 

0) Go to the [MEAD website](https://github.com/uniBruce/Mead) and download the dataset.

1) Resample the video to 25FPS by calling the following: 
```bash 
bash convert_to_25fps.sh <downloaded_dir> <resampled_dir>
```

2) Download the processed data by running:

```bash
download_processed_mead.sh
```
Instead of downloading the processed data, you can also process the dataset yourself. You can find the instructions on that below.

3) Coming soon


# Processing MEAD video data processing 

## Processing stages

This includes the following steps: 
1) Face and landmark detection (mediapipe and FAN)
2) Cropped video extraction 
3) Audio extraction 
4) Segmentation 
5) Emotion feature extraction 
6) Pseudo-GT 3D face reconstruction 

It is better, if these steps are executed separately. Please inspect the [`process_mead.py`](./process_mead.py) script. You will find the following list of variables. 
- `extract_audio` 
- `detect_landmarks` 
- `detect_aligned_landmarks`
- `reconstruct_faces` 
- `recognize_emotions` 
I recommend setting only one to `True` (i.e. `extract_audio`), the others to `False` before procssing all of the dataset's samples. Once the processing pass finishes, 
set the next one to `True`  (i.e. `detect_landmarks`) and the otherw to `False` and make another processing pass. Continue until you've finished all of the above steps. 


If you wish to save time, `extract_audio` and `detect_landmarks` can be ran together. `detect_aligned_landmarks` is better ran separately.  Phases `reconstruct_faces` and `recognize_emotions` can also be ran together. 

## Running all shards in a processing stage 
Each run of the `process_mead.py` script processes one "shard" of data. You can set the variable `videos_per_shard` either in the script or by specifying it: 

```bash
python <input_dir> <output_dir> <shard_index> <number_of_videos_per_shard>
```

`<input_dir>` corresponds to the folder where your MEAD data (resampled to 25 fps lives)
`<output_dir>` is the location where you want the process datasets 