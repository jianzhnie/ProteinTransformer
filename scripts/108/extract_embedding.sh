## af2
python extract_embeddings.py  \
--data_path /data/xbiome/protein_classification \
--pretrain_model_dir /data/xbiome/pre_trained_models/exp4_longformer \
--split "test" \
--batch-size 64

python extract_embeddings.py  \
--data_path /data/xbiome/protein_classification \
--pretrain_model_dir /data/xbiome/pre_trained_models/exp4_longformer \
--split "train" \
--batch-size 64



## 108
python extract_embeddings.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--split "test" \
--batch-size 32

python extract_embeddings.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--split "train" \
--batch-size 64


python extract_ontotextual_embeddings.py \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--tokenizer_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--pretrain_model_dir /home/niejianzheng/xbiome/DeepFold/work_dir/ontotextual_embeddings/checkpoint/checkpoint-1650 \
--embedding_file_name bert_fintune.pkl \
--batch-size 256
