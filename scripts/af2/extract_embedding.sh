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


## extract bert-pretrained model embeddings
python extract_ontotextual_embeddings.py \
--data_path /data/xbiome/protein_classification \
--tokenizer_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--pretrain_model_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  \
--embedding_file_name bert_pretrain.pkl \
--batch-size 64

## extract bert-fintuned model embeddings
python extract_ontotextual_embeddings.py \
--data_path /data/xbiome/protein_classification \
--tokenizer_dir microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--pretrain_model_dir /home/af2/xbiome/DeepFold/work_dir/ontotextual_model/models  \
--embedding_file_name bert_fintune.pkl \
--batch-size 64


## extract seq2vec model embeddings
python extract_seq2vec_embeddings.py \
--data_path /data/xbiome/protein_classification \
--model seq2vec \
--pool_mode sum \
--split "test" \
--batch-size 16
