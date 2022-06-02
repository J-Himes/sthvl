# Test runs on 1 GPU

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_caption.py
