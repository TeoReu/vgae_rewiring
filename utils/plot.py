import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from utils.vgae import test


def create_nx_graph_from_indices(add_self_loops, indices):
    G = nx.Graph()
    if not add_self_loops:
        edges = [(i, j) for i, j in zip(indices[0].tolist(), indices[1].tolist()) if i != j]
        G.add_edges_from(edges)
    else:
        G.add_edges_from(zip(indices[0].tolist(), indices[1].tolist()))
    return G


def plot_paired_graphs(model, data, model_pictures_path, gen_graphs, threshold, add_self_loops):
    graphs = np.random.choice(len(data), gen_graphs, False)

    test_graph_list = []
    for g_id in graphs:
        test_graph_list.append(data[g_id])
    test_loader = DataLoader(test_graph_list)

    adj, gen_adj = test(model,test_loader, gen_graphs)

    for graph in range(gen_graphs):
        recon_adj_binary = gen_adj[graph] > threshold

        adj_binary = adj[graph] != 0
        indices = torch.where(adj_binary.squeeze())
        gen_indices = torch.where(recon_adj_binary)

        G1 = create_nx_graph_from_indices(add_self_loops, indices)
        G2 = create_nx_graph_from_indices(add_self_loops, gen_indices)

        fig, (ax1, ax2) = plt.subplots(1, 2)

        # plot the first graph on the first column
        nx.draw(G1, ax=ax1, with_labels=True)
        ax1.set_title("True Graph")

        # plot the second graph on the second column
        nx.draw(G2, ax=ax2, with_labels=True)
        ax2.set_title("Rewired Graph")

        plt.savefig(model_pictures_path + '/' + str(graph) + '.png', dpi=30)
        # plt.show()
