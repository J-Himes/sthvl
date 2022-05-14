# Test runs on 1 GPU

DATATYPE="youcook"
TRAIN_CSV="data/youcookii/youcookii_train.csv"
VAL_CSV="data/youcookii/youcookii_val.csv"
DATA_PATH="data/youcookii/youcookii_data.no_transcript.pickle"
FEATURES_PATH="data/youcookii/youcookii_videos_features.pickle"
INIT_MODEL="weight/univl.pretrained.bin"
OUTPUT_ROOT="ckpts"

source ~/.bashrc
conda activate sthvl

python -m torch.distributed.launch --nproc_per_node=1 \
main_task_retrieval.py \
--do_train --num_thread_reader=16 \
--epochs=5 --batch_size=32 \
--n_display=100 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_youcook_retrieval --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 64 --visual_num_hidden_layers 6 \
--datatype ${DATATYPE} --init_model ${INIT_MODEL}
