# Test runs on 4 GPUs

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=4 main_task_caption.py
