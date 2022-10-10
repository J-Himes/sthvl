# Test runs on 1 GPU

CONFIG_PATH="conf/local/summ"
CONFIG_NAME="msrvtt_retrieval_dsnet"

source ~/.bashrc
conda activate dsnet

python -m torch.distributed.run --nproc_per_node=1 main_task_retrieval.py ${CONFIG_PATH} ${CONFIG_NAME}
