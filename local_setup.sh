# Make sure to run this file at the top level directory of your project

source ~/.bashrc
conda create -n sthvl python=3.6.9 tqdm boto3 requests pandas --yes
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sthvl

yes | pip install torch==1.10.1
yes | pip install git+https://github.com/Maluuba/nlg-eval.git@master

mkdir -p ./ckpts
mkdir -p ./weight
