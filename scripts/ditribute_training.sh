## local machine
python main.py \
--data_path /Users/robin/xbiome/datasets/protein \
--output-dir /Users/robin/xbiome/DeepFold/work_dir \
--epochs 2 \
--lr 0.001 \
--epochs 10 \
--batch-size 1

#####
python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 ./
python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 ./
python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 --loss-scale 128.0 ./
python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -batch-size 8 --workers 4 --opt-level O2  --loss-scale 128.0
####

# single gpu
python main.py \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--lr 0.001 \
--epochs 10 \
--batch-size 8


## distributed
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  main_amp.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--lr 0.001 \
--epochs 10 \
--batch-size 8 \
--workers 4  \
--opt-level O2  \
--loss-scale 128.0
