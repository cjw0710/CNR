from ogb.nodeproppred import NodePropPredDataset

dataset_name = 'ogbn-arxiv'
dataset = NodePropPredDataset(name=dataset_name)

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
attributes = graph.keys()

# 遍历每个属性，并输出其值
for attribute in attributes:
    print(f"属性名：{attribute}")
    print(f"属性值：{graph[attribute]}")
from collections import deque

# 假设 graph 是一个字典，包含了图的边索引信息
edge_index = graph['edge_index']

# 构建图的邻接表表示
adj_list = {}
for i, j in zip(edge_index[0], edge_index[1]):
    if i not in adj_list:
        adj_list[i] = []
    if j not in adj_list:
        adj_list[j] = []
    adj_list[i].append(j)
    adj_list[j].append(i)

# BFS 计算图的直径
def bfs(start):
    visited = set()
    queue = deque([(start, 0)])
    max_distance = 0
    max_node = start
    
    while queue:
        node, distance = queue.popleft()
        if node not in visited:
            visited.add(node)
            if distance > max_distance:
                max_distance = distance
                max_node = node
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
    return max_node, max_distance

# 随机选择一个节点作为起始点，计算最远节点和距离
start_node = next(iter(adj_list.keys()))
farthest_node, diameter = bfs(start_node)

print("Graph Diameter:", diameter)


# 读取 titleabs.tsv 文件，提取论文 ID 并保存到集合中
# titleabs_paper_ids = set()
# with open('titleabs.tsv', 'r', encoding='utf-8') as f:
#     for line in f:
#         paper_id = line.strip().split('\t')[0]
#         titleabs_paper_ids.add(paper_id)

# # 筛选与 titleabs.tsv 中论文 ID 相同的 filtered_data.txt 文件中的每行内容，并保存到新文件中
# with open('filtered_data.txt', 'r') as f:
#     with open('matched_filtered_data.txt', 'w') as new_f:
#         for line in f:
#             paper_id = line.strip().split('\t')[0]
#             if paper_id in titleabs_paper_ids:
#                 new_f.write(line)

