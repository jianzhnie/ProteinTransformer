from collections import defaultdict
import math
import os

def read_go_children(input_go_obo_file):
    children = defaultdict(list)
    alt_id = defaultdict(list)

    term = False
    go_id = ''
    alt_ids = set()

    with open(input_go_obo_file) as read_in:
        for line in read_in:
            splitted_line = line.strip().split(':')
            if '[Term]' in line:  # new term begins
                term = True
                go_id = ''
                alt_ids = set()

            elif term and 'id: GO:' in line and 'alt_id' not in line:
                go_id = "GO:{}".format(splitted_line[2].strip())
            elif term and 'alt_id: GO' in line:
                alt_id_id = "GO:{}".format(splitted_line[2].strip())
                alt_ids.add(alt_id_id)
                alt_id[go_id].append(alt_id_id)
            elif term and 'is_a:' in line:
                splitted_term = splitted_line[2].split("!")
                go_term = "GO:{}".format(splitted_term[0].strip())
                children[go_term].append(go_id)
                for a in alt_ids:
                    children[go_term].append(a)
            elif '[Typedef]' in line:
                term = False

    return children, alt_id

if __name__ == 'main':
    data_path = '/data/xbiome/protein_classification/'
    all_children, alt_id = read_go_children(os.path.join(data_path, 'go.obo'))
    check = []

    go_ic = {}
    with open(os.path.join(data_path, 'all_GOA_IC.txt')) as lines:
        for line in lines:
            data = line.split('\t')
            ic = float(data[1].replace('\n', ''))
            go_ic[data[0]] = ic

    go_cnt = {}
    with open(os.path.join(data_path, "all_GOA_cnt.txt")) as lines:
        for line in lines:
            data = line.split('\t')
            cnt = float(data[1].replace('\n', ''))
            go_cnt[data[0]] = cnt

    f1 = open(os.path.join(data_path, "all_go_cnt.tsv"), "w")

    with open(os.path.join(data_path, "all_go_edge.txt")) as read_in:
        for line in read_in:
            splitted_line = line.split("\t")
            children = splitted_line[0]
            parent = splitted_line[1].replace("\n", "")
            cnt_every = 0.0
            cnt_chidren = 0.0
            cnt_freq_children = 0.0
            cnt_freq_parent = 0.0
            cnt_freq = 0.0

            if children in go_cnt.keys():
                cnt_freq_children = go_cnt[children]
            if parent in go_cnt.keys():
                cnt_freq_parent = go_cnt[parent]
            if cnt_freq_parent == 0.0 or cnt_freq_children == 0.0:
                cnt_freq = 1.0
            else:
                cnt_freq = cnt_freq_children / cnt_freq_parent

            if parent in all_children:
                for x in all_children[parent]:
                    if x in go_ic.keys():
                        cnt_every += go_ic[x]

            if parent in go_ic.keys():
                cnt_every += go_ic[parent]
            if children in go_ic.keys():
                cnt_chidren += go_ic[children]

            cnt_every += cnt_chidren
            final_cnt = (cnt_chidren / cnt_every) + cnt_freq
            f1.write('{}\t{}\t{}\n'.format(children, parent, cnt_freq))
