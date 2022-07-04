##
python tools/main_multimodal.py \
        --data_path ./data/cafa3 \
        --train_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_train.pkl \
        --val_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
        --namespace mfo \
        --output-dir work_dir  \
        --lr 0.001 \
        --epochs 20 \
        --batch-size 64 \
        --workers 4


##
python tools/main_multimodal.py \
        --data_path ./data/cafa3 \
        --train_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_train.pkl \
        --val_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
        --namespace mfo \
        --output-dir work_dir  \
        --lr 0.01 \
        --epochs 20 \
        --batch-size 256 \
        --workers 4



## inference
