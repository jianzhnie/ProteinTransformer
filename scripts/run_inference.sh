## inference
python inference_embedding.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--resume /home/niejianzheng/xbiome/DeepFold/work_dir/ProtLM_esm_embedding_mean/model_best.pth.tar \
--model esm_embedding \
--pool_mode  mean \
--batch-size  128 \
--workers 4
