import pandas as pd
import os
import sys
sys.path.append('../')
from deepfold.data.utils.ontology import Ontology



def seperate(data_file, go_file):
    df = pd.read_pickle(data_file)
    ont = Ontology(go_file, with_rels=True)
    bpo_proteins = []
    bpo_sequences = []
    bpo_annotations = []
    mfo_proteins = []
    mfo_sequences = []
    mfo_annotations = []
    cco_proteins = []
    cco_sequences = []
    cco_annotations = []
    for item in df.iterrows():
        protien = item[1]['proteins']
        seq = item[1]['sequences']
        annotation = item[1]['annotations']
        bpo_annotation = []
        mfo_annotation = []
        cco_annotation = []
        for term in annotation:
            if ont.get_namespace(term) == 'biological_process':
                bpo_annotation.append(term)
            elif ont.get_namespace(term) == 'molecular_function':
                mfo_annotation.append(term)
            elif ont.get_namespace(term) == 'cellular_component':
                cco_annotation.append(term)
        if len(bpo_annotation) > 0:
            bpo_proteins.append(protien)
            bpo_sequences.append(seq)
            bpo_annotations.append(bpo_annotation)
        if len(mfo_annotation) > 0:
            mfo_proteins.append(protien)
            mfo_sequences.append(seq)
            mfo_annotations.append(mfo_annotation)
        if len(cco_annotation) > 0:
            cco_proteins.append(protien)
            cco_sequences.append(seq)
            cco_annotations.append(cco_annotation)
        
    bpo_df = pd.DataFrame({
        'proteins': bpo_proteins,
        'sequences': bpo_sequences,
        'annotations': bpo_annotations,
    })
    mfo_df = pd.DataFrame({
        'proteins': mfo_proteins,
        'sequences': mfo_sequences,
        'annotations': mfo_annotations,
    })
    cco_df = pd.DataFrame({
        'proteins': cco_proteins,
        'sequences': cco_sequences,
        'annotations': cco_annotations,
    })
    return bpo_df, mfo_df, cco_df



if __name__ == '__main__':
    data_path = '/data/xbiome/protein_classification/cafa3'
    train_data_file = os.path.join(data_path, 'train_data.pkl')
    test_data_file = os.path.join(data_path, 'test_data.pkl')
    terms_file = os.path.join(data_path, 'terms.pkl')
    go_file = os.path.join(data_path, 'go_cafa3.obo')
    bpo_df, mfo_df, cco_df = seperate(train_data_file, go_file)
    save_bpo = os.path.join(data_path, 'bpo')
    save_mfo = os.path.join(data_path, 'mfo')
    save_cco = os.path.join(data_path, 'cco')
    bpo_df.to_pickle(os.path.join(save_bpo, 'bpo_train_data.pkl'))
    mfo_df.to_pickle(os.path.join(save_mfo, 'mfo_train_data.pkl'))
    cco_df.to_pickle(os.path.join(save_cco, 'cco_train_data.pkl'))
    # test data
    bpo_df, mfo_df, cco_df = seperate(test_data_file, go_file)
    bpo_df.to_pickle(os.path.join(save_bpo, 'bpo_test_data.pkl'))
    mfo_df.to_pickle(os.path.join(save_mfo, 'mfo_test_data.pkl'))
    cco_df.to_pickle(os.path.join(save_cco, 'cco_test_data.pkl'))
