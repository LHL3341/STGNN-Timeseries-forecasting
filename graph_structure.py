import torch
import networkx as nx
import matplotlib.pyplot as plt
def construct_graph(nodenum,type):
    if type =='full':
        i_idx = torch.arange(0, nodenum).T.unsqueeze(1).repeat(1, nodenum).flatten().to('cpu').unsqueeze(0)
        j_idx = torch.arange(0, nodenum).repeat(1, nodenum).to('cpu')
        edge_idx = torch.cat((i_idx,j_idx),dim=0)
    if type == 'cor':
        pass
    return edge_idx

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

def show_graph(edge_idx,epoch,dataset,savepath):
    G = nx.Graph()
    G.add_edges_from(edge_idx.T)
    nx.draw(G)
    plt.savefig(f'{savepath}/{dataset}/epoch{epoch}.jpg')

