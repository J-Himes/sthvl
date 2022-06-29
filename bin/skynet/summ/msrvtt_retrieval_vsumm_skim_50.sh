# Test runs on 4 GPUs

CONFIG_PATH="conf/skynet/summ"
CONFIG_NAME="msrvtt_retrieval_vsumm_skim_50"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_retrieval.py ${CONFIG_PATH} ${CONFIG_NAME}
