# Make sure to run this file at the top level directory of your project

conda init bash
source ~/.bashrc
conda create -n sthvl python=3.6.9 tqdm boto3 requests pandas
source ~/.bashrc
conda activate sthvl

pip install torch==1.10.1
pip install git+https://github.com/Maluuba/nlg-eval.git@master

mkdir -p ./ckpts
mkdir -p ./weight
