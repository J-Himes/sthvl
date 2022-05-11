# Make sure to run this file inside your project's main directory.

# Downloads the YouCookII Dataset

mkdir -p data
cd data
wget https://github.com/microsoft/UniVL/releases/download/v0/youcookii.zip
unzip youcookii.zip

# Downloads the MSRVTT Dataset

wget https://github.com/microsoft/UniVL/releases/download/v0/msrvtt.zip
unzip msrvtt.zip
cd ..

# Downloads the Weights
wget -P ./weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
