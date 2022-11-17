
This folder contains pre-training scripts for LERT, which is mainly based on [Google's BERT implementation](https://github.com/google-research/bert). 

The original implementation is based on Tensorflow 1.15 with Cloud TPU devices.

The `run.pretrain.sh` is the starting script (on TPU).

```bash
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
```

### Read This Before Using

1. Linguistic tags used in this paper are listed in lines 110-112 of `run_pretraining.py`, where POS feature has 28 tags, NER feature has 13 tags, and DEP feature has 14 tags. Please do not change the order of them, as they are directly mapped to our released pre-trained weights (if you would like load these weights and perform further pre-training or linguistic prediction).

```python
POS_LIST = ["POS-n", "POS-v", "POS-wp", "POS-u", "POS-d", "POS-a", "POS-m", "POS-p", "POS-r", "POS-ns", "POS-c", "POS-q", "POS-nt", "POS-nh", "POS-nd", "POS-j", "POS-i", "POS-b", "POS-ni", "POS-nz", "POS-nl", "POS-z", "POS-k", "POS-ws", "POS-o", "POS-h", "POS-e", "POS-%"]
NER_LIST = ["NER-O", "NER-S-Ns", "NER-S-Nh", "NER-B-Ni", "NER-E-Ni", "NER-I-Ni", "NER-S-Ni", "NER-B-Ns", "NER-E-Ns", "NER-I-Ns", "NER-B-Nh", "NER-E-Nh", "NER-I-Nh"]
DEP_LIST = ["DEP-ATT", "DEP-WP", "DEP-ADV", "DEP-VOB", "DEP-SBV", "DEP-COO", "DEP-RAD", "DEP-HED", "DEP-POB", "DEP-CMP", "DEP-LAD", "DEP-FOB", "DEP-DBL", "DEP-IOB"]
```

2. To perform linguistically-informed pre-training, please specify the end steps of scaling for each linguistic feature. Line 292-294 in `run_pretraining.py` represents the one used in our paper.
3. **You MUST generate tfrecords yourself, before using this script.** For intellectual reasons, we DO NOT provide scripts for data generation. You can use [BERT original `create_pretraining_data.py` implementation](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) (possibly with this [tutorial](https://github.com/google-research/bert#pre-training-with-bert)) and adjust them to our pre-training task, these includes:
- Perform whole word masking, N-gram masking.
   
- Generate linguistic features for the masked tokens using [LTP](https://github.com/HIT-SCIR/ltp) (or other something similar tool). Once again, please note that if you would like to reuse the pre-trained linguistic head weights, please generate linguistic tags within the one provided above (note #1).
   
- Remove the next sentence prediction (NSP) task.

