# Test runs on 1 GPU

CONFIG_PATH="conf/local"
CONFIG_NAME="msrvtt_caption"

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sthvl

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=0
export MASTER_PORT=60000

python main_task_caption.py ${CONFIG_PATH} ${CONFIG_NAME}
