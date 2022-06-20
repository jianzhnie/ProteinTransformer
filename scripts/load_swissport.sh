#!/bin/bash/
nohup python load_swissport.py \
        --go-file /data/xbiome/protein_classification/go.obo \
        --swissprot-file  /data/xbiome/protein_classification/uniprot_trembl.dat.gz \
        --out-file /data/xbiome/protein_classification/uniprot_trembl.pkl
