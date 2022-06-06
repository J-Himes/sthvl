# Test runs on 4 GPUs

CONFIG_PATH="conf/skynet"
CONFIG_NAME="youcook_caption"

source ~/.bashrc
conda activate sthvl

torchrun main_task_caption.py ${CONFIG_PATH} ${CONFIG_NAME}
