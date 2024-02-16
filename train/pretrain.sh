#!/bin/bash
export PYTHONPATH=/data/code/train
CUDA_VISIBLE_DEVICES=0 python3 /data/code/train/train.py \
	--model_dir "/data/SapBERT-from-PubMedBERT-fulltext" \
	--train_dir "/babeldata/training_data/" \
	--output_dir "/data/SapBERT-fine-tuned-babel" \
	--logging_file "/data/SapBERT-fine-tuned-babel/train_log.txt" \
	--use_cuda \
	--epoch 1 \
	--train_batch_size 256 \
	--learning_rate 2e-5 \
	--max_length 25 \
	--checkpoint_step 17300 \
	--parallel \
	--amp \
	--pairwise \
	--random_seed 33 \
	--loss ms_loss \
	--use_miner \
	--type_of_triplets "all" \
	--miner_margin 0.2 \
	--agg_mode "cls"
