import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, remove_self_loops


def get_new_edge_index(data):
    new_data_list = []
    for graph in data:
        a = graph.edge_index
        b = graph.vr_edge_index
        value_b = graph.vr_edge_weight
        b, value_b = remove_self_loops(b, value_b)
        b_adj = to_dense_adj(b, edge_attr=value_b)
        a_adj = to_dense_adj(a)
        b_adj = b_adj-a_adj
        b_adj[b_adj < 0] = 0
        topk_values, topk_indices = torch.topk(b_adj.flatten(), a.shape[1], largest=True)
        B = torch.zeros_like(b_adj).float()
        B.view(-1)[topk_indices.long()] = topk_values.float()
        B = B.view(b_adj.size())
        new_data_list.append(graph)
        new_edge_index, new_edge_weight = dense_to_sparse(B)
        new_edge_attr = torch.ones(new_edge_weight.shape) * 3
        graph.edge_index = torch.cat([graph.edge_index, new_edge_index], dim=1)
        graph.edge_attr = torch.cat([graph.edge_attr, new_edge_attr], dim=0)
        new_data_list.append(graph)
    return new_data_list

