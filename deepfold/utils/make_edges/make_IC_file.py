import math
import os
from collections import defaultdict


def read_go_children(input_go_obo_file):
    """read 'go.obo' file.

    Args:
        input_go_obo_file (string): go.obo official file

    Returns:
        tuple: children dict and alt_id dict
    """
    children = defaultdict(list)
    alt_id = defaultdict(list)

    term = False
    go_id = ''
    alt_ids = set()

    with open(input_go_obo_file) as read_in:
        for line in read_in:
            splitted_line = line.strip().split(':')
            if '[Term]' in line:
                term = True
                go_id = ''
                alt_ids = set()

            elif term and 'id: GO:' in line and 'alt_id' not in line:
                go_id = 'GO:{}'.format(splitted_line[2].strip())
            elif term and 'alt_id: GO' in line:
                alt_id_id = 'GO:{}'.format(splitted_line[2].strip())
                alt_ids.add(alt_id_id)
                alt_id[go_id].append(alt_id_id)
            elif term and 'is_a:' in line:
                splitted_term = splitted_line[2].split('!')
                go_term = 'GO:{}'.format(splitted_term[0].strip())
                children[go_term].append(go_id)
                for a in alt_ids:
                    children[go_term].append(a)
            elif '[Typedef]' in line:
                term = False

    return children, alt_id


def find_all_descendants(input_go_term, children):
    children_set = set()
    queue = []
    queue.append(input_go_term)
    while queue:
        node = queue.pop(0)
        if node in children and node not in children_set:
            node_children = children[node]
            queue.extend(node_children)
        children_set.add(node)

    return children_set


def store_counts_for_GO_terms(input_go_cnt_file, alt_id):
    """get terms count.

    Args:
        input_go_cnt_file (str): go.obo official file
        alt_id (dict): alt_id dict

    Returns:
        dict: count of all terms
    """
    go_cnt = defaultdict()
    with open(input_go_cnt_file, 'r') as read_in:
        for line in read_in:
            tmp = line.split('\t')
            term = tmp[0]
            cnt = int(tmp[1].replace('\n', ''))
            if term in alt_id.keys():
                for x in alt_id[term]:
                    term = x
                    if term not in go_cnt:
                        go_cnt[term] = cnt
                    else:
                        go_cnt[term] = go_cnt[term] + cnt
            else:
                if term not in go_cnt:
                    go_cnt[term] = cnt
                else:
                    go_cnt[term] = go_cnt[term] + cnt
    return go_cnt


def write_output_file(ic_dict, output_file):
    f = open(output_file, 'w')
    for term in ic_dict.keys():
        ic = ic_dict[term]
        f.write('{}\t{}\n'.format(term, ic))
    f.close()

    return


def calculate_freq(term, children_set, go_cnt):
    """calculate freq(term)

    Args:
        term (str): the Term need to calculate
        children_set (_type_): _description_
        go_cnt (dict): _description_

    Returns:
        int: count of all terms
    """
    freq = 0
    if term in go_cnt.keys():
        freq = freq + go_cnt[term]
    for children in children_set:
        if children in go_cnt.keys():
            freq = freq + go_cnt[children]

    return freq


def calculate_information_contents_of_GO_terms(input_go_cnt_file, children,
                                               alt_id):
    """calculate IC for all terms.

    Args:
        input_go_cnt_file (_type_): _description_
        children (_type_): _description_
        alt_id (_type_): _description_

    Returns:
        dict: dict consist of term:IC value
    """
    ic_dict = defaultdict()
    go_cnt = store_counts_for_GO_terms(input_go_cnt_file, alt_id)

    for x in range(0, 3):
        if x == 0:
            root = 'GO:0005575'  # cellular component
        elif x == 1:
            root = 'GO:0008150'  # biological process
        elif x == 2:
            root = 'GO:0003674'  # molecular function

        root_descendants = find_all_descendants(root, children)
        root_freq = calculate_freq(root, root_descendants, go_cnt)

        for term in root_descendants:
            term_descendants = find_all_descendants(term, children)
            term_freq = calculate_freq(term, term_descendants, go_cnt)
            term_prob = (term_freq + 1) / (root_freq + 1)
            term_ic = -math.log(term_prob)
            assert (term not in ic_dict)
            ic_dict[term] = term_ic

    return ic_dict


if __name__ == 'main':
    data_path = '/data/xbiome/protein_classification/'
    input_go_obo_file = os.path.join(data_path, 'go.obo')
    input_go_cnt_file = os.path.join(data_path, 'all_GOA_cnt.txt')
    output_file = os.path.join(data_path, 'all_GOA_IC.txt')

    children, alt_id = read_go_children(input_go_obo_file)
    ic_dict = calculate_information_contents_of_GO_terms(
        input_go_cnt_file, children, alt_id)
    write_output_file(ic_dict, output_file)

    print('FINISHED\n')
