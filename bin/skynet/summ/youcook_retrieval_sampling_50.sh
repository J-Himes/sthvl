# Test runs on 4 GPUs

CONFIG_PATH="conf/skynet/summ"
CONFIG_NAME="youcook_retrieval_sampling_50"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_retrieval.py ${CONFIG_PATH} ${CONFIG_NAME}