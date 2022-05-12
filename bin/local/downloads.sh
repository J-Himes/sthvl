# Make sure to run this file inside your project's main directory.

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sthvl

# Downloads the Bert Model

mkdir modules/bert-base-uncased
cd modules/bert-base-uncased/
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv bert-base-uncased-vocab.txt vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
tar -xvf bert-base-uncased.tar.gz
rm bert-base-uncased.tar.gz
cd ../../

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
