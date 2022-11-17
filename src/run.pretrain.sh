#!/bin/bash
set -ex

TPU_NAME="your-tpu-name"
TPU_ZONE="your-tpu-zone"
DATA_DIR=./your-path-to-tfrecords
MODEL_DIR=./your-path-to-model-saving
CONFIG_FILE=./your-path-to-config-file

# run pretraining
python run_pretraining.py \
	--input_file=${DATA_DIR}/tf_examples.tfrecord.* \
	--output_dir=${MODEL_DIR} \
	--do_train=True \
	--bert_config_file=${CONFIG_FILE} \
	--train_batch_size=1024 \
	--eval_batch_size=1024 \
	--max_seq_length=512 \
	--max_predictions_per_seq=75 \
	--num_train_steps=2000000 \
	--num_warmup_steps=10000 \
	--save_checkpoints_steps=50000 \
	--learning_rate=1e-4 \
	--do_lower_case=True \
	--use_tpu=True \
	--tpu_name=${TPU_NAME} \
	--tpu_zone=${TPU_ZONE}