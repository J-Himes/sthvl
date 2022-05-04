# Make sure to run this file at the top level directory of your project

conda create -n sthvl python=3.6.9 tqdm boto3 requests pandas
source ~/.bashrc
conda activate sthvl
pip install torch==1.10.2
pip install git+https://github.com/Maluuba/nlg-eval.git@master

ln -s /srv/share/datasets/avsd data
ln -s /srv/essa-lab/share3/avsd/ckpts ckpts
ln -s /srv/essa-lab/share3/avsd/modules/bert-base-uncased modules/bert-base-uncased
ln -s /srv/essa-lab/share3/avsd/weight weight 
