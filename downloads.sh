# Make sure to run this file at the top level directory of your project

# Downloads the YouCookII Dataset

mkdir -p data
cd data
wget https://github.com/microsoft/UniVL/releases/download/v0/youcookii.zip
unzip youcookii.zip

# Downloads the MSRVTT Dataset

wget https://github.com/microsoft/UniVL/releases/download/v0/msrvtt.zip
unzip msrvtt.zip
cd ..
