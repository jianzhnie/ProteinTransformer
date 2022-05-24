## inference
python inference.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--resume /home/niejianzheng/xbiome/DeepFold/work_dir/ProtLM_esm/checkpoint_0.pth \
--model esm \
--pool_mode pooler \
--batch-size 16
