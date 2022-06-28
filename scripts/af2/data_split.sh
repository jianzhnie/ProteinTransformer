#!/bin/bash
python deepfold/data/data_split.py \
    --go-file data/go.obo \
    --data-file data/data/swissprot.pkl \
    --out-terms-file data/data/terms.pkl \
    --train-data-file data/data/train_data.pkl \
    --test-data-file data/data/test_data.pkl
