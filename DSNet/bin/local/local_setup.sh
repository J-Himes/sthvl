# Make sure to run this file from the DSNet directory

source ~/.bashrc
conda create -n dsnet python=3.6 tqdm boto3 requests pandas --yes
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
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

mkdir -p datasets/ && cd datasets/
wget https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip
unzip dsnet_datasets.zip
rm dsnet_datasets.zip
cd ..

mkdir -p models && cd models
wget https://www.dropbox.com/s/0jwn4c1ccjjysrz/pretrain_ab_basic.zip
unzip pretrain_ab_basic.zip
rm pretrain_ab_basic.zip

wget https://www.dropbox.com/s/2hjngmb0f97nxj0/pretrain_af_basic.zip
unzip pretrain_af_basic.zip
rm pretrain_af_basic.zip

cd ..
echo
echo To test setup, move to the src directory.
echo
echo From src, run ./test_eval.sh to test evaluation on pre-trained models.
echo
echo Then run ./test_training.sh to test anchor-based training.
echo
