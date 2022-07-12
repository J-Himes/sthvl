# Test runs on 4 GPUs

CONFIG_PATH="conf/skynet"
CONFIG_NAME="pretrain_stage_one"

source ~/.bashrc
conda activate sthvl_hydra

torchrun main_pretrain.py ${CONFIG_PATH} ${CONFIG_NAME}
