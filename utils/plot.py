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


def plot_paired_graphs(model, data, model_pictures_path, gen_graphs, threshold, add_self_loops, device):
    graphs = np.random.choice(len(data), gen_graphs, False)

    test_graph_list = []
    for g_id in graphs:
        test_graph_list.append(data[g_id])
    test_loader = DataLoader(test_graph_list)

    adj, gen_adj = test(model,test_loader, gen_graphs, device)

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

        plt.savefig(model_pictures_path + '/' + str(graph) + '.png', dpi=60)
        # plt.show()

def plot_paired_graph_weighted_edges(model, data, path, threshold, device):
    G1 = nx.Graph()
    G2 = nx.Graph()

    data_test = data[0:5]
    test_loader = DataLoader(data_test)

    adj, gen_adj = test(model,test_loader, 5, device)
    for graph in range(5):
        A1 = adj[graph].cpu().detach().squeeze().numpy()
        A2 = gen_adj[graph].cpu().detach().squeeze().numpy()

        for i in range(A1.shape[0]):
            for j in range(A1.shape[1]):
                if A1[i][j] > threshold:
                    G1.add_edge(i, j, weight=A1[i][j])

        for i in range(A2.shape[0]):
            for j in range(A2.shape[1]):
                if A2[i][j] > threshold and i != j:
                    G2.add_edge(i, j, weight=A2[i][j])

        fig, (ax1, ax2) = plt.subplots(1, 2)

        edge_widths1 = [G1[u][v]['weight'] * G1[u][v]['weight'] for u, v in G1.edges()]
        edge_widths2 = [(G2[u][v]['weight']+ 0.2) * (0.2 + G2[u][v]['weight']) for u, v in G2.edges()]


        # plot the first graph on the first column
        nx.draw(G1,  width=edge_widths1, ax=ax1, with_labels=True)
        ax1.set_title("True Graph")

        # plot the second graph on the second column
        nx.draw(G2, width=edge_widths2, ax=ax2, with_labels=True)
        ax2.set_title("Rewired Graph")

        plt.savefig(path + '/' + str(graph) + '.png', dpi=100)

