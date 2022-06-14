## evaluate diamond
python  evaluate_diamondscore.py \
    --train-data-file /Users/robin/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /Users/robin/xbiome/datasets/protein/test_data.pkl \
    --diamond-scores-file /Users/robin/xbiome/datasets/protein/test_diamond.res \
    --ontology-obo-file /Users/robin/xbiome/datasets/protein/go.obo


## evaluate model
python  evaluate_deepmodel.py \
    --train-data-file /Users/robin/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /Users/robin/xbiome/datasets/protein/test_data.pkl \
    --terms-file /Users/robin/xbiome/datasets/protein/terms.pkl \
    --ontology-obo-file /Users/robin/xbiome/datasets/protein/go.obo \
    --output_dir /home/niejianzheng/xbiome/DeepFold/work_dir



## af2
python  evaluate_deepmodel.py \
    --train-data-file /data/xbiome/protein_classification/train_data.pkl \
    --test-data-file /data/xbiome/protein_classification/predictions.pkl \
    --terms-file /data/xbiome/protein_classification/terms.pkl \
    --ontology-obo-file /data/xbiome/protein_classification/go.obo \
    --output_dir /home/af2/xbiome/DeepFold/work_dir



## 108
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

## evaluate go_sim
python  evaluate_gosim.py \
    --train-data-file /home/niejianzheng/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /home/niejianzheng/xbiome/datasets/protein/predictions.pkl \
    --gosim-scores-file /home/niejianzheng/xbiome/goPredSim/results/all_predicts.txt \
    --ontology-obo-file /home/niejianzheng/xbiome/datasets/protein/go.obo \
    --output_dir /home/niejianzheng/xbiome/DeepFold/work_dir
