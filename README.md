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



## Protein-Language Model

- **Deep generative models of genetic variation capture the effects of mutations**

- **Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences**

- **Evaluating protein transfer learning with tape**

- **Large-scale clinical interpretation of genetic variants using evolutionary data and deep learning**



## Protein function Annotation

- **DeepGO: predicting protein functions from sequence and interactions using a deep ontology-aware classifier** [[link\]](https://dx.doi.org/10.1093/bioinformatics/btx624)

- **Deep Semantic Protein Representation for Annotation, Discovery, and Engineering** [[link\]](https://www.biorxiv.org/content/early/2018/07/10/365965)

- **Embeddings from deep learning transfer GO annotations beyond homology**
- **Improving Protein Function Annotation via Unsupervised Pre-training: Robustness, Efficiency, and Insights**

- **Using Deep Learning to Annotate the Protein Universe** [[link\]](https://www.biorxiv.org/content/early/2019/05/03/626507)





## Structure Prection

- **End-to-end differentiable learning of protein structure**
- **Improved protein structure prediction using potentials from deep learning**
- **Improved protein structure prediction using predicted interresidue orientations** [[link\]](https://www.pnas.org/content/117/3/1496)
- **Improved protein structure prediction using predicted interresidue orientations**
- **Energy-based models for atomic-resolution protein conformations**
- **MSA Transformer**



## Protein Generation

- **Machine-learning-guided directed evolution for protein engineering**
- **Machine learning in enzyme engineering**
- **Low-N protein engineering with data-efficient deep learning**
- **Progen: Language modeling for protein generation**
- **ProtTrans: towards cracking the language of Life's code through self-supervised deep learning and high performance computing**
- **De novo protein design by deep network hallucination**
- **Deep diversification of an AAV capsid protein by machine learning**



## Evaluation

- **A large-scale evaluation of computational protein function prediction**
