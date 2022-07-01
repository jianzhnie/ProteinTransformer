diamond makedb --in data/train_data.fa -d data/train_data #creates train_data.dmnd

diamond blastp  -d data/train_data.dmnd --more-sensitive -t /tmp -q data/test_data.fa --outfmt 6 qseqid sseqid bitscore -o data/test_diamond.res
