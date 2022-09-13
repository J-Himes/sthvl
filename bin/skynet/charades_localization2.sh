# Test runs on 4 GPUs

CONFIG_PATH="conf/skynet"
CONFIG_NAME="charades_localization2"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_localization.py ${CONFIG_PATH} ${CONFIG_NAME}
