# Test runs on 4 GPUs

DATATYPE="msrvtt"
TRAIN_CSV="data/data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/data/msrvtt/MSRVTT_data.json"
FEATURES_PATH="data/data/msrvtt/msrvtt_videos_features.pickle"
INIT_MODEL="weights/weight/univl.pretrained.bin"
OUTPUT_ROOT="ckpts"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py \
--do_train --num_thread_reader=16 \
--epochs=5 --batch_size=128 \
--n_display=100 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_retrieval --bert_model bert-base-uncased \
--do_lower_case --lr 5e-5 --max_words 48 --max_frames 48 \
--batch_size_val 64 --visual_num_hidden_layers 6 \
--datatype ${DATATYPE} --expand_msrvtt_sentences --init_model ${INIT_MODEL}
