# Make sure to run this file at the top level directory of your project

source ~/.bashrc
conda create -n sthvl python=3.6 tqdm boto3 requests pandas --yes
source ~/.bashrc
conda activate sthvl

yes | pip install hydra-core --upgrade
yes | pip install git+https://github.com/Maluuba/nlg-eval.git@master
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch --yes

rm data
rm ckpts
rm modules/bert-base-uncased
rm weights
ln -s /coc/dataset/sthvl data
ln -s /coc/pskynet3/sthvl/ckpts ckpts
ln -s /coc/pskynet3/sthvl/modules/bert-base-uncased modules/bert-base-uncased
ln -s /coc/pskynet3/sthvl/weights weights
