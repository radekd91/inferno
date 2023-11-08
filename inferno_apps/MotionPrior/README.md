# Motion Prior

 ![emotion_recog](flint.png)

This project accompanies the release of [EMOTE](../TalkingHead/). A temporal, transformer-based VAE called FLINT, is an essential component of EMOTE. 

## Pretrained model
The pretrained model used in EMOTE is publicly available. Go to [EMOTE](../TalkingHead/demos/) and run `download_assets.sh` to download it.


## Training your own FLINT

If you want to train your own FLINT, follow the data processing instructions in [EMOTE's data processing](../TalkingHead/data_processing/). 

Then run the following: 
```bash 
python training/train_flint.py
```
Please refer to [`training/train_flint.py`](./training/train_flint.py) for additional settings.


## Contribute with your own Motion Prior 
If you design a new motion prior, please create a PR. We will merge it. :-) 



## Citation 

If you use this work in your publication, please cite the following:
```
@inproceedings{EMOTE,
  title = {Emotional Speech-Driven Animation with Content-Emotion Disentanglement},
  author = {Daněček, Radek and Chhatre, Kiran and Tripathi, Shashank and Wen, Yandong and Black, Michael and Bolkart, Timo},
  publisher = {ACM},
  month = dec,
  year = {2023},
  doi = {10.1145/3610548.3618183},
  url = {https://emote.is.tue.mpg.de/index.html},
  month_numeric = {12}
}
```
