#!/bin/bash

DEFEAULT_PATH=${PWD}/"../../assets/data/mead_25fps"
## The first argument is the path to the folder where the data will be downloaded (if there is any)

if [ $# -eq 0 ]; then
    echo "No arguments supplied, using default path: ${DEFEAULT_PATH}"
    DATA_PATH=${DEFEAULT_PATH}
else
    echo "Using path: $1"
    DATA_PATH=$1
fi

mkdir -p ${DATA_PATH}
cd ${DATA_PATH}
mkdir -p processed
cd processed

# Download the processed data
echo "Downloading the processed data to ${DATA_PATH}/processed. This might take a while."
echo "Downloading the metadata..."
# if the file does not exist, download it
if [ ! -f metadata.pkl ]; then
    wget https://download.is.tue.mpg.de/emote/mead_25fps/processed/metadata.pkl -O metadata.pkl
else
    echo "metadata.pkl already exists, skipping download"
fi

echo "Downloading the detected bounding boxes..."
if [ ! -f detections.zip ]; then
    wget https://download.is.tue.mpg.de/emote/mead_25fps/processed/detections.zip -O detections.zip
else
    echo "detections.zip already exists, skipping download"
fi

echo "Downloading the detected ladmarks..."
if [ ! -f landmarks.zip ]; then
    wget https://download.is.tue.mpg.de/emote/mead_25fps/processed/landmarks.zip -O landmarks.zip
else
    echo "landmarks.zip already exists, skipping download"
fi

echo "Downloading the recognized emotions..."
if [ ! -f emotions.zip ]; then
    wget https://download.is.tue.mpg.de/emote/mead_25fps/processed/emotions.zip -O emotions.zip
else
    echo "emotions.zip already exists, skipping download"
fi

# echo "Downloading the reconstructions used in the EMOTE paper"
# if [ ! -f reconstruction_v0.zip ]; then
#     wget https://download.is.tue.mpg.de/emote/mead_25fps/processed/reconstruction_v0.zip -O reconstruction_v0.zip
# else
#     echo "reconstruction_v0.zip already exists, skipping download"
# fi

echo "Downloading the newer version of the reconstructions..."
if [ ! -f reconstruction_v1.zip ]; then
    wget https://download.is.tue.mpg.de/emote/mead_25fps/processed/reconstruction_v1.zip -O reconstruction_v1.zip
else
    echo "reconstruction_v1.zip already exists, skipping download"
fi

echo "Processed data downloaded successfully."

## Unzip the downloaded files
echo "Unzipping the downloaded files. This might take a while."

echo "Unzipping the metadata..."
unzip -q metadata.pkl.zip
echo "Unzipping the detected bounding boxes..."
unzip -q detections.zip
echo "Unzipping the detected ladmarks..."
unzip -q landmarks.zip
echo "Unzipping the recognized emotions..."
unzip -q emotions.zip
# echo "Unzipping the reconstructions used in the EMOTE paper"
# unzip -q reconstruction_v0.zip
echo "Unzipping the newer version of the reconstructions..."
unzip -q reconstruction_v1.zip

echo "Data unzipped succsessfully."

