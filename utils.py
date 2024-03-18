import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops

def knn_graph(X, k=20, metric='minkowski'):
    X = X.cpu().detach().numpy()
    A = kneighbors_graph(X, n_neighbors=k, metric=metric)
    edge_index = sparse_mx_to_edge_index(A)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index

# def kmeans_graph(X, k):
#     # 使用 KMeans 对数据进行聚类
#     kmeans = KMeans(n_clusters=k)
#     labels = kmeans.fit_predict(X.cpu().detach().numpy())

#     # 获取每个簇的中心点
#     cluster_centers = torch.tensor(kmeans.cluster_centers_)

#     # 计算节点之间的相似度（欧氏距离）
#     similarity_matrix = torch.cdist(X, cluster_centers)

#     # 根据相似度矩阵定义边的连接
#     threshold = similarity_matrix.mean()  # 可以根据需要调整阈值
#     edges = (similarity_matrix < threshold).nonzero(as_tuple=False).t()

#     # 构建边索引
#     edge_index, _ = remove_self_loops(edges)

#     return edge_index
def kmeans_graph(X, k):
    # 使用 KMeans 对数据进行聚类
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)

    # 获取每个簇的中心点
    cluster_centers = torch.tensor(kmeans.cluster_centers_).float()

    # 计算节点之间的相似度（欧氏距离）
    similarity_matrix = torch.cdist(torch.tensor(X, dtype=torch.float), cluster_centers)

    # 根据相似度矩阵定义边的连接
    threshold = similarity_matrix.mean()  # 可以根据需要调整阈值
    edges = (similarity_matrix < threshold).nonzero(as_tuple=False).t()

    # 构建边索引
    edge_index, _ = remove_self_loops(edges)

    return edge_index



def sparse_mx_to_edge_index(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    row = torch.from_numpy(sparse_mx.row.astype(np.int64))
    col = torch.from_numpy(sparse_mx.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
 
    return edge_index

if __name__ == '__main__':
    path = osp.join(osp.expanduser('~'), 'data', 'cora')
    dataset = Planetoid(path, 'cora')
    data = dataset[0]
    knn_graph = knn_graph(data.x)
    print(knn_graph.size())