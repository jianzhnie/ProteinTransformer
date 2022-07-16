## mfo
python tools/main_multimodal.py \
        --data_path ./protein_data/cafa3 \
        --train_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_train.pkl \
        --val_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
        --namespace mfo \
        --output-dir work_dir/ \
        --lr 0.001 \
        --epochs 1 \
        --batch-size 64 \
        --workers 4


## bpo
python tools/main_multimodal.py \
        --data_path ./protein_data/cafa3 \
        --train_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_train.pkl \
        --val_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
        --namespace mfo \
        --output-dir work_dir  \
        --lr 0.01 \
        --epochs 20 \
        --batch-size 256 \
        --workers 4

## cco
python tools/main_multimodal.py \
        --data_path /home/niejianzheng/xbiome/datasets/protein/cafa3/process \
        --train_file_name cco_esm1b_t33_650M_UR50S_embeddings_mean_train.pkl \
        --val_file_name cco_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
        --namespace cco \
        --output-dir work_dir \
        --lr 0.001 \
        --epochs 10 \
        --batch-size 64 \
        --workers 4



##

python tools/main_multimodal_bert.py \
        --data_path /home/niejianzheng/xbiome/datasets/protein/cafa3/process \
        --train_file_name cco_esm1b_t33_650M_UR50S_embeddings_mean_train.pkl \
        --val_file_name cco_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
        --namespace cco \
        --output-dir work_dir \
        --lr 0.001 \
        --epochs 10 \
        --batch-size 64 \
        --workers 4
