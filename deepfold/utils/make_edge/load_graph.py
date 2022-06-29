import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp


def load_edge_list(edge_list_dir,terms_embedding_file,terms_file,symmetrize=False):
    terms_embedding = pd.read_pickle(terms_embedding_file)
    terms_df = pd.read_pickle(terms_file)
    terms_df = terms_df.rename(columns={"terms":"go"})
    terms_embedding = terms_embedding.merge(terms_df,on='go')
    terms_df = terms_embedding.loc[:,['go']]
    terms_df = terms_df.rename(columns={"go":"0"})
    df = pd.read_csv(edge_list_dir, header=None, sep='\t', engine='c')
    df.dropna(inplace=True)
    df = df.rename(columns={0:"0",1:"1",2:"2"})
    df = df.merge(terms_df,on='0')
    terms_df = terms_df.rename(columns={"0":"1"})
    df = df.merge(terms_df,on='1')
    if symmetrize:
            rev = df.copy().rename(columns={0: 1, 1: 0})
            df = pd.concat([df, rev])
        # pd.factorize 输入一个序列，返回一个元组。
        # 元组的第一个位置是 输入序列的index序列形式
        # 元组的第二个位置是 输入序列的unique形式，也就是为输入序列构造了一个映射表。
    idx, objects = pd.factorize(df[['0', '1']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    IC = df['2'].astype('float')
    return idx, objects.tolist(), IC.tolist(), terms_df

def load_node_list(node_list_dir, terms_df):
    terms_only_num = set(term[3:] for term in terms_df.terms.values)
    with open(node_list_dir, "r") as read_line:
        go_id = []
        for line in read_line:
            splitted_line = line.strip().split('\t')
            s,t = str(splitted_line[0]), str(splitted_line[1])
            if s in terms_only_num and t in terms_only_num:
                tmp = [s, t]
                go_id.append(tmp)
    return go_id

def normalize(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj
def sparse_mx_to_torch_sparse_tensor(sparse_adj):
    sparse_adj = sparse_adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_adj.row, sparse_adj.col)).astype(np.int64))
    values = torch.from_numpy(sparse_adj.data)
    shape = torch.Size(sparse_adj.shape)
    return torch.sparse.LongTensor(indices, values, shape)

def build_adj(idx, IC, idx_map):
    adj_new = []
    for i in range(len(idx_map)):
        tmp = [0] * (len(idx_map))
        adj_new.append(tmp)

    for i in range(len(idx)):
        adj_new[idx[i][0]][idx[i][1]] = IC[i]
        adj_new[idx[i][1]][idx[i][0]] = IC[i]
    adj_new = np.array(adj_new)
    return adj_new

def build_graph(terms_file):
    edge_list_dir = "all_go_cnt.tsv"
    node_list_dir = "all_go_only_num.tsv"
    idx, objects, IC, terms_df = load_edge_list(edge_list_dir,go_embedding_file,terms_file,symmetrize=False)
    terms_df = terms_df.rename(columns={"1":"terms"})
    go_id = load_node_list(node_list_dir, terms_df)
    go_id = np.array(go_id)
    go_id = go_id.reshape(-1)
    # remove duplicate values
    go_id = pd.unique(go_id).tolist()
    idx_2d = idx

    edges = np.array(idx)
    labels = np.array(objects)
    idx = idx.reshape(-1)

    tmp = []
    idx = pd.unique(idx).tolist()
    for i in range(len(go_id)):
        go_id[i] = format(int(go_id[i]), '07')
        tmp.append("GO:%07d" % (int(go_id[i])))
    # making dictionary
    idx_map = dict(zip(go_id, idx))
    label_map = dict(zip(tmp, idx))
    ldx_map_ivs = dict(zip(idx, go_id))
    label_map_ivs = dict(zip(idx, tmp))
    
    terms_df = pd.DataFrame({"go":label_map.keys()})
    terms_embedding = pd.read_pickle(go_embedding_file)
    terms_df = terms_df.merge(terms_embedding,on='go')
    
    if embedding_type == 'mean_embedding':
        terms_embedding = torch.Tensor(list(terms_df.mean_embedding.values))
    
    # build symmetric adjacency matrix
    adj = build_adj(idx_2d, IC, idx_map)
    adj = sp.coo_matrix(adj)
    adj = adj + np.multiply(adj.T, adj.T > adj) - np.multiply(adj, (adj.T > adj))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    return adj, terms_embedding, label_map, label_map_ivs