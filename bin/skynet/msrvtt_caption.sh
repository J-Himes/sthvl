# Test runs on 1 GPU

CONFIG_PATH="conf/skynet"
CONFIG_NAME="msrvtt_caption"

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_caption.py ${CONFIG_PATH} ${CONFIG_NAME}
