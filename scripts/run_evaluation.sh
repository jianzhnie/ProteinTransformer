## evaluate diamond
python  evaluate_diamondscore.py \
    --train-data-file /Users/robin/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /Users/robin/xbiome/datasets/protein/test_data.pkl \
    --diamond-scores-file /Users/robin/xbiome/datasets/protein/test_diamond.res \
    --ontology-obo-file /Users/robin/xbiome/datasets/protein/go.obo


## evaluate model
python  evaluate_.py \
    --train-data-file /Users/robin/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /Users/robin/xbiome/datasets/protein/test_data.pkl \
    --terms-file /Users/robin/xbiome/datasets/protein/terms.pkl \
    --ontology-obo-file /Users/robin/xbiome/datasets/protein/go.obo \
    --output_dir //Users/robin/xbiome/
