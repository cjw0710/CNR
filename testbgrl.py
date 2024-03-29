import torch
import random
import numpy as np
import os
import dgl
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.data import citation_graph as citegrh
import networkx as nx
import matplotlib.pyplot as plt


seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


data = citegrh.load_cora()

# # 加载学习到的嵌入
# embeds = torch.load('embeds.pt', map_location=device)
# print(f'embeds: {len(embeds)}')
# embeds_cpu = embeds.cpu().detach().numpy()

# # 计算余弦相似度矩阵
# cosine_sim_matrix = cosine_similarity(embeds_cpu, embeds_cpu)

# # 获取相似度大于0.95的节点对的索引
# indices = np.where(cosine_sim_matrix > 0.95)

# # 提取节点对及其对应的相似度值
# similar_pairs = list(zip(indices[0], indices[1], cosine_sim_matrix[indices]))

# # 对节点对进行排序以确保一致的节点顺序
# sorted_similar_pairs = [tuple(sorted(pair[:2])) + (pair[2],) for pair in similar_pairs]

# # 删除重复记录
# unique_similar_pairs = list(set(sorted_similar_pairs))

# 假设 data[0].ndata['feat'] 的形状为 (num_nodes, num_features)
num_nodes, num_features = data[0].ndata['feat'].shape


g = dgl.DGLGraph().to(device)
g.add_nodes(num_nodes)


src_nodes, dst_nodes = data[0].edges()
src_nodes = src_nodes.to(device)
dst_nodes = dst_nodes.to(device)
g.add_edges(src_nodes, dst_nodes)
num_edges1 = g.num_edges()
# for pair in unique_similar_pairs:
#     i, j, _ = pair
    
#     # Check if the edge already exists before adding
#     if i != j:
#         g.add_edges([i], [j])
# num_edges2 = g.num_edges()
# added_edges = num_edges2 - num_edges1
# print("图中增加边的数量:", added_edges)


g_cpu = g.cpu()
nx_g = dgl.to_networkx(g_cpu)
nx_g = nx_g.to_undirected()


plt.figure(figsize=(10, 10))
nx.draw(nx_g, with_labels=True)
plt.savefig('graph.png')  
plt.show()

g = dgl.add_self_loop(g)


g.ndata['feat'] = torch.FloatTensor(data[0].ndata['feat']).to(device)
g.ndata['label'] = torch.LongTensor(data[0].ndata['label']).to(device)

class BGRL(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout_rate):
        super(BGRL, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)  # 使用GraphConv
        self.fc1 = nn.Linear(in_feats, hidden_size)  # 输入特征大小到隐藏层大小的线性变换层
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 隐藏层大小到输出类别数量的线性变换层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.dropout(x)
        x = self.fc2(x)  # 将隐藏层大小转换为输出类别数量
        return F.log_softmax(x, dim=1)  # 对输出进行log_softmax处理，以便在训练时使用CrossEntropyLoss损失函数




in_feats = data[0].ndata['feat'].shape[1]
hidden_size = 256
num_classes = len(torch.unique(data[0].ndata['label']))
dropout_rate = 0.7


model = BGRL(in_feats, hidden_size, num_classes, dropout_rate).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()


train_mask = torch.BoolTensor(data[0].ndata['train_mask']).to(device)
val_mask = torch.BoolTensor(data[0].ndata['val_mask']).to(device)
test_mask = torch.BoolTensor(data[0].ndata['test_mask']).to(device)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(g, g.ndata['feat'])
    loss = criterion(output[train_mask], g.ndata['label'][train_mask])
    loss.backward()
    optimizer.step()

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

f_acc = []


for r in range(5):  
    all_test_accuracies = []  
    for experiment in range(100): 
        model = BGRL(in_feats, hidden_size, num_classes, dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss() 
        best_val_accuracy = 0.0
        best_epoch = 0
        early_stop_counter = 0
        max_early_stop_counter = 10
        test_accuracy = 0.0
        
       
        for epoch in range(200):
            train(epoch)
            val_accuracy = evaluate(model, g, g.ndata['feat'], g.ndata['label'], val_mask)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                early_stop_counter = 0
                torch.save(model.state_dict(), 'best_model.pkl4')
            # else:
            #     early_stop_counter += 1
            #     if early_stop_counter >= max_early_stop_counter:
            #         print(f"Early stopping at epoch {epoch}")
            #         break

        best_model = BGRL(in_feats, hidden_size, num_classes, dropout_rate).to(device)
        best_model.load_state_dict(torch.load('best_model.pkl4'))
        best_model.eval()
            
        with torch.no_grad():
            test_logits = best_model(g, g.ndata['feat'])
            
        _, test_indices = torch.max(test_logits[test_mask], dim=1)
        test_correct = torch.sum(test_indices == g.ndata['label'][test_mask])
        test_accuracy = test_correct.item() / len(g.ndata['label'][test_mask])
        all_test_accuracies.append(test_accuracy*100)
        
        print(f'Experiment {experiment + 1}, Test Accuracy: {test_accuracy*100}')
    
    f_acc.append(np.mean(all_test_accuracies))

mean_f_acc = np.mean(f_acc)
std_f_acc = np.std(f_acc)

print(f'\nMean Test Accuracy: {mean_f_acc:.4f}')
print(f'Standard Deviation Test Accuracy: {std_f_acc:.4f}')
