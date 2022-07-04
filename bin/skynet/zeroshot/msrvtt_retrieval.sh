# Test runs on 4 GPUs

CONFIG_PATH="conf/skynet/zeroshot"
CONFIG_NAME="msrvtt_retrieval"

source ~/.bashrc
conda activate sthvl

torchrun main_task_retrieval.py ${CONFIG_PATH} ${CONFIG_NAME}
