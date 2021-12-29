#!/bin/bash

echo "------------------------------"
echo "TRAIN GBCNET"
echo "------------------------------"

python train.py \
	--train_set_name="train_set.txt" \
	--test_set_name="val_set.txt" \
	--load_model \
	--epochs=30 \
	--lr=0.003 \
	--load_path="weights/init_weights.pth" \
	--save_dir="outs/gbcnet"

echo "------------------------------"
echo "TRAIN GBCNET w/ CURRICULUM"
echo "------------------------------"

python train.py \
	--train_set_name="train_set.txt" \
	--test_set_name="val_set.txt" \
	--load_model \
	--load_path="weights/init_weights.pth" \
	--save_dir="outs/gbcnet-va" \
	--va
