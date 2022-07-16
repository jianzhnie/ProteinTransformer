## inference
python tools/inference_multimodal.py  \
--data_path ./protein_data/cafa3 \
--test_file_name mfo_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--resume /home/niejianzheng/xbiome/DeepFold/work_dir/prot_gcn/ProtLM_protgcn/model_best.pth.tar \
--namespace mfo \
--batch-size  128 \
--workers 4



## cco
python tools/inference_multimodal.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein/cafa3/process \
--test_file_name cco_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--resume /home/niejianzheng/xbiome/DeepFold/work_dir/ProtLM_protgcn/model_best.pth.tar \
--namespace cco \
--batch-size  128 \
--workers 4



## cco
python tools/inference_multimodal_bert.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein/cafa3/process \
--test_file_name cco_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--resume /home/niejianzheng/xbiome/DeepFold/work_dir/ProtLM_protgcn/model_best.pth.tar \
--namespace cco \
--batch-size  128 \
--workers 4
