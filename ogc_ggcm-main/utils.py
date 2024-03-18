import sys
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torch.nn import Module

class LinearNeuralNetwork(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    """
    def __init__(self, nfeat, nclass, bias=True):
        super(LinearNeuralNetwork, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=bias) # Y=W*X if bias=false; else Y=W*X+b

    def forward(self, x):
        return self.W(x)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    # 加载学习到的嵌入
    embeds = torch.load('../embeds.pt', map_location='cuda')
    print(f'embeds: {len(embeds)}')
    embeds_cpu = embeds.cpu().detach().numpy()
    
    # 计算余弦相似度矩阵
    cosine_sim_matrix = cosine_similarity(embeds_cpu, embeds_cpu)
    
    # 获取相似度大于0.85的节点对的索引
    indices = np.where(cosine_sim_matrix > 0.99)
    
    # 提取节点对及其对应的相似度值
    similar_pairs = list(zip(indices[0], indices[1], cosine_sim_matrix[indices]))
    
    # 对节点对进行排序以确保一致的节点顺序
    sorted_similar_pairs = [tuple(sorted(pair[:2])) + (pair[2],) for pair in similar_pairs]
    
    # 删除重复记录
    unique_similar_pairs = list(set(sorted_similar_pairs))
    # 计算邻接矩阵中值为1的元素个数
    num_edges = np.count_nonzero(adj.A)

    
    print("Number of edges in adj:", num_edges)

    # 添加相似节点对到邻接矩阵中
    for i, j, _ in unique_similar_pairs:
        # 如果节点不是相同节点并且之间没有边，则添加边
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1  # 或者根据需要设置其他值，如相似度
            # 计算邻接矩阵中值为1的元素个数
    num_edges = np.count_nonzero(adj.A)

        
    print("Number of edges in adj:", num_edges)
    features = row_normalize(features)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def use_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    return device