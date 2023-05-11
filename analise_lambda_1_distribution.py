import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader

from models.vgae import L1VGAE, VariationalEncoderwithModel
from utils.dataset import split_dataset, split_dataset_peptides
from utils.spectral import pos_eigenvalues, first_pos_eigenvalue
from torch_geometric.utils import degree

from utils.vgae import test


def lambda_one_adj(adj):
    return first_pos_eigenvalue(adj)


def lambda_one_list_adj(adj_list):
    return [lambda_one_adj(adj) for adj in adj_list]

def plot_lambda_distribution(dataset, model, layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == "ZINC":
        train_set, test_set, val_set = split_dataset('no')
        edge_attr_dim = 1
    else:
        train_set, test_set, val_set = split_dataset_peptides('no')
        edge_attr_dim = train_set[0].edge_attr.shape[1]

    deg = None
    if model == "PNA":
        max_degree = -1
        for data in train_set:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_set:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

    in_channels, out_channels, lr, n_epochs = train_set[0].num_features, 200, 0.001, 20

    gen_graphs, threshold, batch_size, add_self_loops = 3, 0.65, 20, False

    vae = L1VGAE(VariationalEncoderwithModel(in_channels=in_channels, out_channels=out_channels, layers=4,
                                               molecular=False, transform='no', model=model, deg=deg,
                                               edge_dim=edge_attr_dim), device)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    for alpha in ["-10.0", "-5.0","0.0", "0.01", "0.1", "1.0", "2.0", "5.0"]:
        vae.load_state_dict(torch.load(
            'vgae/zinc/model_PNA/layers_4/transform_no/alpha_'+ alpha +'/model.pt', map_location=torch.device('cpu')))

        adj, gen_adj =  test(vae, test_loader, gen_graphs, device)

        for i in range(len(gen_adj)):
            gen_adj[i] = (gen_adj[i] >= 0.5).float().unsqueeze(dim=0)

        color = np.random.rand(3,)

        e1_recon_adj = lambda_one_list_adj(gen_adj.squeeze())

        plt.hist(np.array(e1_recon_adj), alpha=0.5, label=alpha, color=color)

    plt.legend(loc='upper right')
    plt.show()
    # Plot the mean line with the same color as the histogram

    return 0

plot_lambda_distribution("ZINC", "PNA", 4)