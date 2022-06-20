#!/bin/bash/
nohup python load_swissport.py \
        --go-file /data/xbiome/protein_classification/go.obo \
        --swissprot-file  /data/xbiome/protein_classification/uniprot_trembl_split_0 \
        --out-file /data/xbiome/protein_classification/uniprot_trembl_0.pkl

sleep 5h
nohup python load_swissport.py \
        --go-file /data/xbiome/protein_classification/go.obo \
        --swissprot-file  /data/xbiome/protein_classification/uniprot_trembl_split_1 \
        --out-file /data/xbiome/protein_classification/uniprot_trembl_1.pkl

sleep 10h
nohup python load_swissport.py \
        --go-file /data/xbiome/protein_classification/go.obo \
        --swissprot-file  /data/xbiome/protein_classification/uniprot_trembl_split_2 \
        --out-file /data/xbiome/protein_classification/uniprot_trembl_2.pkl

sleep 15h
nohup python load_swissport.py \
        --go-file /data/xbiome/protein_classification/go.obo \
        --swissprot-file  /data/xbiome/protein_classification/uniprot_trembl_split_3 \
        --out-file /data/xbiome/protein_classification/uniprot_trembl_3.pkl

sleep 20h
nohup python load_swissport.py \
        --go-file /data/xbiome/protein_classification/go.obo \
        --swissprot-file  /data/xbiome/protein_classification/uniprot_trembl_split_4 \
        --out-file /data/xbiome/protein_classification/uniprot_trembl_4.pkl
