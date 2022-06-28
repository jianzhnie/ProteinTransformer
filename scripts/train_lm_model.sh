# 108
python main_ontotextual_embeddings.py \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--tokenizer_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--pretrain_model_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir/ontotextual_model  \
--epochs 20 \
--batch-size 64

# af2
python main_ontotextual_embeddings.py \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--tokenizer_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--pretrain_model_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir/ontotextual_model  \
--epochs 20 \
--batch-size 64
