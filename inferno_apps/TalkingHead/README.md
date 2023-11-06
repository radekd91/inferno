# EMOTE : Emotional Speech-Driven Animation with Content-Emotion Disentanglement

This is the official implementation of [EMOTE : Emotional Speech-Driven Animation with Content-Emotion Disentanglement](https://emote.is.tue.mpg.de/). 
EMOTE takes speech audio and an emotion and intensity labels on the input and produces a talking head avatar that correctly articulates the words spoken in the audio while expressing the specified emotion.


## Installation 

1) Follow the steps at the [root of this repo](../..). If for some reason the environment from there is not valid, create one using a `.yml` file from `envs`.

2) In order to run the demos you will need to download and unzip a few assets. Run `download_assets.sh` to do that: 
```bash 
cd demos 
bash download_assets.sh
```
3) (Optional for inference, required for training) [Basel Face Model](https://faces.dmi.unibas.ch/bfm/bfm2019.html) texture space adapted to FLAME. Unfortunately, we are not allowed to distribute the texture space, since the license does not permit it. Therefore, please go to the [BFM page](https://faces.dmi.unibas.ch/bfm/bfm2019.html) sign up and dowload BFM. Then use the tool from this [repo](https://github.com/TimoBolkart/BFM_to_FLAME) to convert the texture space to FLAME. Put the resulting texture model file file into [`../../assets/FLAME/texture`](../../assets/FLAME/texture) as `FLAME_albedo_from_BFM.npz`


## Demos 

Then activate your environment: 
```bash
conda activate work38
```

### Create a speech-driven animation for all of 8 basic emotions
If you want to run EMOTE on the demo audio file, run the following:

```bash 
python demo/demo_eval_talking_head_on_audio.py 
```

The script will save the output meshes and videos into `.results/` for each of the 9 basic emotions.

To run the demo on any audio, run:
```bash 
python demo/demo_eval_talking_head_on_audio.py --path_to_audio <your_wav_file> --output_folder <your_output_folder>
```
If you only want results for a particular emotion, specify `--emotion` followed by one of: `Neutral`, `Happy`, `Sad`, `Surprise`, `Fear`, `Disgust`,`Anger`, `Contempt`. 
You may also specify more of them and seperate them with `,`.


## Training 
Details coming soon...

<!-- 
Training EMOTE is a multi-step process which consists of: 

1) MEAD data processing 
    - including pseudo-GT extraction
    - for more information about this step go to [data processing](./data_processing/)

2) Training video emotion classifier on MEAD 
    - this classifier predicts the emotion and intensity label on MEAD videos 
    - an sequence-aggregated emotion feature is a byproduct of the classifier and it will be used for the emotion loss in training EMOTE 
    - for more information about this step go to the [Video Emotion Recognition project](../VideoEmotionRecognition/)

3) Training the FLINT Motion Prior 
    - the motion prior is a critical component of EMOTE. It is impossible to apply perceptual losses and not get uncanny artifacts without it
    - for more information, go to the [Motion Prior project](../MotionPrior/)

4) Training the first stage of EMOTE 
    - training only with vertex error loss

5) Training the second stage of EMOTE 
    - finetuning the previous stage with neural renderering, with perceptual losses, with content-emotion disentanglement mechanism
 -->
