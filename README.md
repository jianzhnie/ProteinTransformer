# DeepFold

DeepFold is an ToolKit Using AI for Computing Biology

## Train model

## Gpu Training

```sh
python main.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--lr 0.0001 \
--epochs 10 \
--batch-size 2 \
--log_wandb \
--workers 4
```

### Distributed Training

```sh
torchrun --nnodes=1 --nproc_per_node=2  --rdzv_id=0 main.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--lr 0.0001 \
--epochs 10 \
--batch-size 2 \
--log_wandb \
--workers 4
```
