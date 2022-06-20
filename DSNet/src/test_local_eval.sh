# Make sure to run this file from the src directory

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dsnet

echo
echo Testing evaluation with pretrained models.
echo

python evaluate.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
python evaluate.py anchor-free --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4

echo
echo The anchor-based model should return F-scores of 0.6205 and 0.5019
echo
echo The anchor-free model should return F-scores of 0.6186 and 0.5118
