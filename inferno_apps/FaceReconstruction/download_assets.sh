cd ./../.. 
mkdir -p assets 
cd assets

echo "In order to run FaceReconstruction, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "If you wish to use FaceReconstruction, please register at:" 
echo -e '\e]8;;https://emote.is.tue.mpg.de\ahttps://emote.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emote.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


echo "Downloading assets to run FaceReconstruction..." 

# if FLAME is not downloaded, download it
if [ ! -d "FLAME" ]; then
    echo "Downloading FLAME..."
    wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip -O FLAME.zip
    echo "Extracting FLAME..."
    unzip FLAME.zip
    echo "Assets for FaceReconstruction downloaded and extracted."
else 
    echo "FLAME already downloaded."
fi

# if MICA is not downloaded, download it
if [ ! -d "MICA" ]; then
    echo "Downloading MICA..."
    mkdir -p MICA/model
    wget -O MICA/model/mica.tar "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1"
    echo "Assets for FaceReconstruction downloaded and extracted."
else 
    echo "MICA already downloaded."
fi

echo "Downloading FaceReconstruction models..."
wget https://download.is.tue.mpg.de/emote/FaceReconstruction.zip 
echo "Extracting FaceReconstruction models..."
unzip -n FaceReconstruction.zip


mkdir data 
cd data
echo "Downloading example test data"
wget https://download.is.tue.mpg.de/emoca/assets/data/EMOCA_test_example_data.zip -O EMOCA_test_example_data.zip
unzip EMOCA_test_example_data.zip
echo "Example test data downloaded and extracted."
