
CONFIG_PATH="conf/skynet/dsnet"
CONFIG_NAME="msrvtt_retrieval_dsnet_50"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.run --nproc_per_node=1 main_task_retrieval.py ${CONFIG_PATH} ${CONFIG_NAME}
