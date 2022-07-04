# Test runs on 4 GPUs

CONFIG_PATH="conf/skynet/zeroshot"
CONFIG_NAME="msrvtt_caption"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_caption.py ${CONFIG_PATH} ${CONFIG_NAME}
