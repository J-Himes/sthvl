# Test runs on 4 GPUs

CONFIG_PATH="conf/skynet/summ"
CONFIG_NAME="msrvtt_retrieval_vsumm_key_50"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_caption.py ${CONFIG_PATH} ${CONFIG_NAME}
