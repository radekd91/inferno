cd ../../.. 
mkdir -p assets 
cd assets

# echo "Downloading assets to run Emotion Recognition" 
# wget https://owncloud.tuebingen.mpg.de/index.php/s/WHjQE7t8BE4Re56/download -O EmotionRecognition.zip
# echo "Extracting  Emotion Recognition models"
# unzip EmotionRecognition.zip

echo "Downloading assets to run Emotion Recognition" 

mkdir -p EmotionRecognition/face_reconstruction_based 
cd EmotionRecognition/face_reconstruction_based

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/facerec_based_models/EMOCA-emorec.zip -O EMOCA-emorec.zip
echo "Extracting  EMOCA-emorec"
unzip EMOCA-emorec.zip

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/facerec_based_models/EMOCA_detail-emorec.zip -O EMOCA_detail-emorec.zip
echo "Extracting EMOCA_detail-emorec"
unzip EMOCA_detail-emorec.zip

cd .. 

mkdir -p image_based_networks 
cd image_based_networks

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/ResNet50.zip -O ResNet50.zip 
echo "Extracting ResNet 50"
unzip ResNet50.zip

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/SWIN-B.zip -O SWIN-B.zip
echo "Extracting SWIN B"
unzip SWIN-B.zip
cd ..

echo "Assets for  Emotion Recognition downloaded and extracted."
cd ../../../gdl_apps/EmotionRecognition/demos
