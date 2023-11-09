cd ../.. 
mkdir -p assets 
cd assets

echo "In order to run EMOTE, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "If you wish to use EMOTE, please register at:" 
echo -e '\e]8;;https://emote.is.tue.mpg.de\ahttps://emote.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emote.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "Downloading assets to run EMOTE..." 

echo "Downloading FLAME related assets"
wget https://download.is.tue.mpg.de/emote/FLAME.zip -O FLAME.zip
echo "Extracting FLAME..."
## unzip without overwriting existing files
unzip -n FLAME.zip

echo "Downloading FLINT"
wget https://download.is.tue.mpg.de/emote/MotionPrior.zip 
echo "Extracting FLINT..."
unzip -n MotionPrior.zip

echo "Downloading static emotion feature extractor" 

mkdir -p EmotionRecognition/image_based_networks 
cd EmotionRecognition/image_based_networks 
wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/ResNet50.zip -O ResNet50.zip 
echo "Extracting ResNet 50"
unzip -n ResNet50.zip
cd ../..

echo "Downloading Video Emotion Recognition net"
wget https://download.is.tue.mpg.de/emote/VideoEmotionRecognition.zip 
echo "Extracting Video Emotion Recognition net"
unzip -n VideoEmotionRecognition.zip

echo "Downloading EMOTE..."
wget https://download.is.tue.mpg.de/emote/TalkingHead.zip
echo "Extracting EMOTE..."
unzip -n TalkingHead.zip

echo "Assets for EMOTE downloaded and extracted."

mkdir data 
cd data
echo "Downloading example test data"
wget https://download.is.tue.mpg.de/emote/EMOTE_test_example_data.zip -O EMOTE_test_example_data.zip
unzip -n EMOTE_test_example_data.zip
echo "Example test data downloaded and extracted."
cd ../../inferno_apps/TalkingHead/demos
