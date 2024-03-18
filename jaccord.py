import torch
import dgl
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import pairwise_distances_chunked
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.data import citation_graph as citegrh

# 载入PubMed数据集
data = citegrh.load_cora()  # 修改这里
# 加载学习到的 embeds
embeds = torch.load('embeds.pt')
print(f'embeds: {len(embeds)}')
# 将 embeds 转移到 CPU 上
embeds_cpu = embeds.cpu().detach().numpy()

# 计算 Jaccard 相似度矩阵
jaccard_sim_matrix = 1 - pairwise_distances(embeds.cpu().detach().numpy(), metric='hamming')

# 获取相似度大于0.8的节点对的索引
indices = np.where(jaccard_sim_matrix > 0.85)

# 提取节点对及其对应的相似度值
similar_pairs = list(zip(indices[0], indices[1], jaccard_sim_matrix[indices]))

# 对节点对进行排序，确保节点顺序一致
sorted_similar_pairs = [tuple(sorted(pair[:2])) + (pair[2],) for pair in similar_pairs]

# 移除重复记录
unique_similar_pairs = list(set(sorted_similar_pairs))

print(f" {len(unique_similar_pairs)}")

# 假设 data[0].ndata['feat'] 的形状为 (num_nodes, num_features)
num_nodes, num_features = data[0].ndata['feat'].shape

# 创建一个新图，确保节点数量正确
g = dgl.DGLGraph()
g.add_nodes(num_nodes)

# 添加原始图的边
src_nodes, dst_nodes = data[0].edges()
g.add_edges(src_nodes, dst_nodes)
count = 0

# 根据 unique_similar_pairs 添加边
for pair in unique_similar_pairs:
    i, j, _ = pair
    g.add_edges([i], [j])

# 设置节点特征和标签
g.ndata['feat'] = torch.FloatTensor(data[0].ndata['feat'])
g.ndata['label'] = torch.LongTensor(data[0].ndata['label'])

# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GraphConv(hidden_size, num_classes)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.dropout1(x)
        x = self.conv2(g, x)
        x = self.dropout2(x)
        return x

# 初始化并训练GCN模型
in_feats = data[0].ndata['feat'].shape[1]
hidden_size = 256
num_classes = len(torch.unique(data[0].ndata['label']))
dropout_rate = 0.5  # 可以根据需要调整dropout率
model = GCN(in_feats, hidden_size, num_classes, dropout_rate)

# 划分数据集
train_mask = torch.BoolTensor(data[0].ndata['train_mask'])
val_mask = torch.BoolTensor(data[0].ndata['val_mask'])
test_mask = torch.BoolTensor(data[0].ndata['test_mask'])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()


early_stop_counter = 0
max_early_stop_counter = 10
test_accuracy = 0.0

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(g, g.ndata['feat'])
    loss = criterion(output[train_mask], g.ndata['label'][train_mask])
    loss.backward()
    optimizer.step()
    # print(f'Epoch: {epoch}, Loss: {loss.item()} ')

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        accuracy = correct.item() / len(labels)
        return accuracy
    
num_experiments = 100
all_test_accuracies = []
# 训练模型
for experiment in range(num_experiments):
    best_val_accuracy = 0.0
    best_epoch = 0
    for epoch in range(200):
        train(epoch)
    
        # 在每20轮后开始进行早停判断
        # if epoch >= 20:
        # 在验证集上评估模型
        val_accuracy = evaluate(model, g, g.ndata['feat'], g.ndata['label'], val_mask)
        
        # print(f'Epoch: {epoch}, Validation Accuracy: {val_accuracy}')

        # 检查是否提升了验证集精度
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            early_stop_counter = 0
            # 保存在验证集上性能最好的模型参数
            torch.save(model.state_dict(), 'best_model.pkl')
        else:
            early_stop_counter += 1
    
            # 判断是否达到早停条件
            # if early_stop_counter >= max_early_stop_counter:
            #     print(f'Early stopping at epoch {epoch}...')
            #     break
    
    # 创建相同的模型结构
    best_model = GCN(in_feats, hidden_size, num_classes, dropout_rate)
    
    # 加载保存的最佳模型参数
    best_model.load_state_dict(torch.load('best_model.pkl'))
    
    # 将模型切换到评估模式
    best_model.eval()
    
    # 应用模型到测试集
    with torch.no_grad():
        test_logits = best_model(g, g.ndata['feat'])
    
    # 在测试集上获取预测结果
    _, test_indices = torch.max(test_logits[test_mask], dim=1)
    test_correct = torch.sum(test_indices == g.ndata['label'][test_mask])
    test_accuracy = test_correct.item() / len(g.ndata['label'][test_mask])
    all_test_accuracies.append(test_accuracy)
    print(f'Experiment {experiment + 1}, Test Accuracy: {test_accuracy}')
# 计算平均值和标准差
mean_test_accuracy = np.mean(all_test_accuracies)
std_test_accuracy = np.std(all_test_accuracies)

print(f'\nMean Test Accuracy: {mean_test_accuracy:.4f}')
print(f'Standard Deviation Test Accuracy: {std_test_accuracy:.4f}')
