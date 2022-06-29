import os
from collections import Counter

import pandas as pd


def statistic_terms(dataframe):
    """get frequency of each term in 'prop_annotations'."""
    cnt = Counter()
    for i, row in dataframe.iterrows():
        for term in row['prop_annotations']:
            cnt[term] += 1

    print('Number of prop_annotations:', len(cnt))
    sorted_by_freq_tuples = sorted(cnt.items(), key=lambda x: x[0])
    sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)

    return sorted_by_freq_tuples


# 写入文件
def write_output_file(ic_dict, output_file):
    f = open(output_file, 'w')
    for term in ic_dict.keys():
        ic = ic_dict[term]
        f.write('{}\t{}\n'.format(term, ic))
    f.close()

    return


if __name__ == 'main':
    data_path = '/data/xbiome/protein_classification/'
    swissprot_file = os.path.join(data_path, 'swissprot.pkl')
    df = pd.read_pickle(swissprot_file)
    freq = statistic_terms(df)
    freq_dict = {go: count for go, count in freq}
    write_output_file(freq_dict, os.path.join(data_path, 'all_GOA_cnt.txt'))
