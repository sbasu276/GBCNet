#!/bin/bash

echo "GBCNET"
python test.py \
	--test_set_name="test.txt" \
	--patch=0.15 \
	--load_path="weights/gbcnet.pth"

echo "---------------------------"

echo "GBCNET+Curriculum"
python test.py \
	--test_set_name="test.txt" \
	--patch=0.17 \
	--load_path="weights/gbcnet_va.pth"
