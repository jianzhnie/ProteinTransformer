#!/bin/bash
python main.py \
        --data_path /Users/robin/xbiome/datasets/protein \
        --output-dir /Users/robin/xbiome/DeepFold/work_dir \
        --dataset_name protseq \
        --num_labels 5874 \
        --model esm_embedding \
        --lr 0.001 \
        --epochs 20 \
        --batch-size 2 \
        --workers 4
