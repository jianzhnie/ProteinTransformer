import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser


def read_fasta(file_path):
    """
    id : Unique identifier given each sequence in the dataset.
    sequence : Protein sequence. Each character is seperated by a "space". Will be useful for BERT tokernizer.
    _sequence_length_ : Character length of each protein sequence.
    location : Classification given each sequence.
    _is_train_ : Indicates whether the record be used for training or test. Will be used to seperate the dataset for traning and validation.
    """

    with open(file_path) as fasta_file:  # Will close handle cleanly
        records = []
        for title, sequence in SimpleFastaParser(fasta_file):
            record = []
            title_splits = title.split(None)
            record.append(title_splits[0])  # First word is ID
            sequence = ' '.join(sequence)
            record.append(sequence)
            record.append(len(sequence))
            location_splits = title_splits[1].split('-')
            record.append(location_splits[0])  # Second word is Location
            record.append(location_splits[1])  # Second word is Membrane

            if (len(title_splits) > 2):
                record.append(0)
            else:
                record.append(1)

            records.append(record)
    columns = [
        'id', 'sequence', 'sequence_length', 'location', 'membrane', 'is_train'
    ]
    return pd.DataFrame(records, columns=columns)


if __name__ == '__main__':

    data = read_fasta('./tmp/deeploc_data.fasta', )
    data.head()
