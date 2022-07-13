# Make sure to run this file from the DSNet directory

source ~/.bashrc
conda create -n dsnet python=3.6 tqdm boto3 requests pandas --yes
source ~/.bashrc
conda activate dsnet

yes | pip install hydra-core --upgrade
yes | pip install git+https://github.com/Maluuba/nlg-eval.git@master
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch --yes
conda install h5py --yes
python -m pip install --upgrade --user ortools

pip install opencv-python
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-geometric

rm datasets
rm models

ln -s /coc/dataset/sthvl/data/dsnet datasets
ln -s /coc/pskynet3/sthvl/models/dsnet models
