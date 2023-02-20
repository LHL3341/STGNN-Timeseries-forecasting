import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入GCN层、GraphSAGE层和GAT层
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,GATv2Conv
from torch_geometric.datasets import Planetoid

# 加载数据，出错可自行下载，解决方案见下文
#dataset = Planetoid(root='./tmp/Cora', name='Cora')

class GAT_NET(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=4,dropout=0.2):
        super(GAT_NET, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=4,dropout=dropout)  # 定义GAT层，使用多头注意力机制
        self.gat2 = GATConv(hidden*heads, classes,dropout=dropout)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.BatchNorm1d(hidden*heads)
    def forward(self, x,edge_index):
        x = self.layernorm(self.gat1(x, edge_index))
        x = F.leaky_relu(x)
        x = self.gat2(x, edge_index)
        return x


"""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT_NET(dataset.num_node_features, 16, dataset.num_classes, heads=4).to(device)  # 定义GAT
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('GAT Accuracy: {:.4f}'.format(acc))"""