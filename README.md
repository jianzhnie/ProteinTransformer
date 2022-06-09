# DeepFold

DeepFold is an ToolKit Using AI for Computing Biology

[Toc]

## Installation

The sources for AutoTabular can be downloaded from the `Github repo`.

You can either clone the public repository:

```bash
# clone project
git clone
# First, install dependencies
pip install -r requirements.txt
```

Once you have a copy of the source, you can install it with:

```bash
python setup.py install
```

## How to use

### Single Gpu Training

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

### Evaluate

```sh
## evaluate diamond
python  evaluate_diamondscore.py \
    --train-data-file /home/niejianzheng/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /home/niejianzheng/xbiome/datasets/protein/test_data.pkl \
    --diamond-scores-file /home/niejianzheng/xbiome/datasets/protein/test_diamond.res \
    --ontology-obo-file /home/niejianzheng/xbiome/datasets/protein/go.obo \
    --output_dir /home/niejianzheng/xbiome/DeepFold/work_dir


## evaluate model
python  evaluate_deepmodel.py \
    --train-data-file /home/niejianzheng/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /home/niejianzheng/xbiome/datasets/protein/predictions.pkl \
    --terms-file /home/niejianzheng/xbiome/datasets/protein/terms.pkl \
    --ontology-obo-file /home/niejianzheng/xbiome/datasets/protein/go.obo \
    --output_dir /home/niejianzheng/xbiome/DeepFold/work_dir
```

### Inference

```sh
## inference
python inference_embedding.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--resume /home/niejianzheng/xbiome/DeepFold/work_dir/ProtLM_esm_embedding_mean/model_best.pth.tar \
--model esm_embedding \
--pool_mode  mean \
--batch-size  128 \
--workers 4
```

### Extract Embeddings

```sh
python extract_embeddings.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein \
--split "test" \
--batch-size 32
```

## License

This library is licensed under the Apache 2.0 License.

## Contributing to DeepFold

We are actively accepting code contributions to the AutoTabular project. If you are interested in contributing to AutoTabular, please contact me.
