# af2
python main_ontotextual_embeddings.py \
--data_path /data/xbiome/protein_classification \
--tokenizer_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--pretrain_model_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  \
--output-dir /home/af2/xbiome/DeepFold/work_dir/ontotextual_model  \
--epochs 20 \
--batch-size 128
