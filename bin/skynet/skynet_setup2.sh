source ~/.bashrc
conda create -n sthvl python=3.6 tqdm boto3 requests pandas --yes
source ~/.bashrc
conda activate sthvl

yes | pip install hydra-core --upgrade
yes | pip install git+https://github.com/Maluuba/nlg-eval.git@master
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch --yes


