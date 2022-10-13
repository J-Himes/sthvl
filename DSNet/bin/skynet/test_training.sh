# Make sure to run this file from the DSNet directory

cd src

source ~/.bashrc
conda activate dsnet

echo
echo Testing training with anchor-based model.
echo

python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml
