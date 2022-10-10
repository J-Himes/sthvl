# Test runs on 1 GPU

CONFIG_PATH="conf/local/summ"
CONFIG_NAME="msrvtt_caption_sampling_50"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_caption.py ${CONFIG_PATH} ${CONFIG_NAME}
