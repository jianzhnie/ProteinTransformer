#!/bin/bash

# 108
python main_contrastive.py \
        --data_path /home/niejianzheng/xbiome/datasets/protein \
        --output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
        --dataset_name protseq \
        --num_labels 5874 \
        --model contrastive_lstm \
        --lr 0.001 \
        --epochs 20 \
        --batch-size 64 \
        --workers 4


python main.py \
        --data_path /home/niejianzheng/xbiome/datasets/protein \
        --output-dir /home/niejianzheng/xbiome/DeepFold/work_dir  \
        --dataset_name esm_embedding \
        --num_labels 5874 \
        --model esm_embedding \
        --lr 0.001 \
        --epochs 10 \
        --batch-size 256 \
        --workers 4


python main.py \
        --data_path /home/niejianzheng/xbiome/datasets/protein \
        --output-dir /home/niejianzheng/xbiome/DeepFold/work_dir  \
        --dataset_name esm \
        --num_labels 5874 \
        --model esm \
        --lr 0.001 \
        --epochs 10 \
        --batch-size 4 \
        --workers 4
