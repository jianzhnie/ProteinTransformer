#!/bin/bash

python function_predcit.py \
        --data-path /Users/robin/xbiome/DeepFold/data/data \
        --model-path /Users/robin/xbiome/DeepFold/work_dir/model/deepgoplus \
        --summary-path /Users/robin/xbiome/DeepFold/work_dir/logs/deepgoplus


python function_predcit.py \
        --data-path /home/niejianzheng/xbiome/DeepFold/data/data \
        --model-path /home/niejianzheng/xbiome/DeepFold/work_dir/model/deepgoplus \
        --summary-path home/niejianzheng/xbiome/DeepFold/work_dir/logs/deepgoplus



nohup python protlm.py > results.log 2>&1 &
