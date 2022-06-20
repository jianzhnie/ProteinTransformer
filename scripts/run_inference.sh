## inference
python inference_embedding.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--resume /home/niejianzheng/xbiome/DeepFold/work_dir/ProtLM_esm_embedding_mean/model_best.pth.tar \
--model esm_embedding \
--pool_mode  mean \
--batch-size  128 \
--workers 4


## inference
python inference_protlm.py  \
--data_path /data/xbiome/protein_classification \
--tokenizer_model_dir /data/xbiome/pre_trained_models/exp4_longformer \
--output-dir /home/af2/xbiome/DeepFold/work_dir \
--pretrain_model_dir /home/af2/xbiome/DeepFold/work_dir/protlm_roberta/models  \
--batch-size  64 \
--workers 4

## inference
python inference.py  \
--data_path /data/xbiome/protein_classification \
--tokenizer_model_dir /data/xbiome/pre_trained_models/exp4_longformer \
--output-dir /home/af2/xbiome/DeepFold/work_dir \
--pretrain_model_dir /home/af2/xbiome/DeepFold/work_dir/protlm_roberta/models  \
--batch-size  64 \
--workers 4


python tools/inference.py  \
--data_path /data/xbiome/protein_classification \
--output-dir /home/af2/xbiome/DeepFold/work_dir \
--dataset_name bert_embedding \
--model bert_embedding \
--resume /home/af2/xbiome/DeepFold/work_dir/ProtLM_bert_embedding_mean/model_best.pth.tar
