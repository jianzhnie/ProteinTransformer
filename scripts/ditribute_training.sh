# ## local machine
# python main.py \
# --data_path /Users/robin/xbiome/datasets/protein \
# --output-dir /Users/robin/xbiome/DeepFold/work_dir \
# --epochs 2 \
# --lr 0.001 \
# --epochs 10 \
# --batch-size 1

# # single gpu
python main.py \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--lr 0.001 \
--epochs 10 \
--batch-size 2

## distributed
nohup  python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  main.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--lr 0.0001 \
--epochs 10 \
--batch-size 2 \
--workers 4  > results.log 2>&1 &

## distributed
nohup torchrun --nnodes=1 --nproc_per_node=2  --rdzv_id=0 main.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--lr 0.0001 \
--epochs 20 \
--batch-size 1 \
--log_wandb \
--workers 4 > results.log 2>&1 &
