##


python main_multimodal.py \
        --data_path /data/xbiome/protein_classification/cafa3 \
        --train_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_train.pkl \
        --val_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
        --namespace mfo \
        --output-dir /home/af2/xbiome/DeepFold/work_dir  \
        --lr 0.001 \
        --epochs 10 \
        --batch-size 32 \
        --workers 4
