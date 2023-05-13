import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader

from models.vgae import L1VGAE, VariationalEncoderwithModel
from utils.dataset import split_dataset, split_dataset_peptides
from utils.spectral import pos_eigenvalues, first_pos_eigenvalue
from torch_geometric.utils import degree, dense_to_sparse, get_laplacian, to_dense_adj

from utils.vgae import test

plt.style.use("seaborn-paper")  # Replace "ggplot" with the desired style

def lamda_1_for_weighted_graphs(a):
    edge_index, edge_weight = dense_to_sparse(a)

    l_edge_index, l_edge_weight = get_laplacian(edge_index, edge_weight, "sym", num_nodes=a.shape[0])

    L = to_dense_adj(edge_index=l_edge_index, edge_attr=l_edge_weight, batch_size=1)
    eigenvalues = torch.sort(torch.real(torch.linalg.eigvals(L.squeeze(0))))
    first_pos_eigenvalue = eigenvalues[0][1]

    return first_pos_eigenvalue

def lambda_one_adj(adj):
    return first_pos_eigenvalue(adj)


def lambda_one_list_adj(adj_list):
    return [lamda_1_for_weighted_graphs(adj.squeeze()) for adj in adj_list]

def plot_lambda_distribution(dataset, model, set):
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

    gen_graphs, threshold, batch_size, add_self_loops = 10, 0.5, 1, False

    vae = L1VGAE(VariationalEncoderwithModel(in_channels=in_channels, out_channels=out_channels, layers=4,
                                               molecular=False, transform='no', model=model, deg=deg,
                                               edge_dim=edge_attr_dim), device)
    if set == "train":
        test_loader = DataLoader(train_set, batch_size=batch_size)
    else:
        test_loader = DataLoader(test_set, batch_size=batch_size)

    color_palette = plt.cm.Set2(range(7))
    j = 0
    for alpha in ["-5.0","-1.0", "0.0", "0.1", "1.0","5.0"]:
        alpha_index = hash(alpha) % len(color_palette)

        vae.load_state_dict(torch.load(
            'vgae/'+dataset+'/model_'+ model +'/layers_4/transform_no/alpha_'+ alpha +'/model.pt', map_location=torch.device('cpu')))

        adj, gen_adj = test(vae, test_loader, gen_graphs, device)

        for i in range(len(gen_adj)):
            gen_adj[i] = gen_adj[i].float().unsqueeze(dim=0)



        e1_recon_adj = lambda_one_list_adj(gen_adj)

        plt.hist(np.array(e1_recon_adj), alpha=0.6, label=alpha, color = color_palette[j])
        mean_value = np.mean(e1_recon_adj)
        plt.axvline(mean_value, linestyle='--', color = color_palette[j], label=f"{alpha} mean")
        j += 1

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
              ncol=3, fancybox=True, shadow=True)
    plt.xlabel("Lambda 1")
    plt.ylabel("Frequency")
    title = plt.title("Peptides on " + set + " set - Lambda 1 distribution for different alpha values")
    title.set_position([0.5, 2.05])
    plt.savefig(dataset+"_"+model+"_"+set+"_set_lambda_1_distribution.png", dpi=300)
    plt.show()
    # Plot the mean line with the same color as the histogram

    return 0

def plot_lambda_mean_over_thresholds(dataset, model, set, weighted):
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

    gen_graphs, threshold, batch_size, add_self_loops = 10, 0.5, 1, False

    vae = L1VGAE(VariationalEncoderwithModel(in_channels=in_channels, out_channels=out_channels, layers=4,
                                             molecular=False, transform='no', model=model, deg=deg,
                                             edge_dim=edge_attr_dim), device)
    if set == "train":
        test_loader = DataLoader(train_set, batch_size=batch_size)
    else:
        test_loader = DataLoader(test_set, batch_size=batch_size)
    thresholds = [0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    color_palette = plt.cm.Set2(range(7))
    j = 0
    for alpha in ["-5.0", "-1.0", "0.0", "0.1", "1.0", "5.0"]:
        alpha_index = hash(alpha) % len(color_palette)
        threshold_means = []
        for threshold in thresholds:
            vae.load_state_dict(torch.load(
                'vgae/' + dataset + '/model_' + model + '/layers_4/transform_no/alpha_' + alpha + '/model.pt',
                map_location=torch.device('cpu')))

            adj, gen_adj = test(vae, test_loader, gen_graphs, device)

            for i in range(len(gen_adj)):
                if weighted == False:
                    gen_adj[i] = (gen_adj[i]>=threshold).float().unsqueeze(dim=0)
                else:
                    gen_adj[i] = gen_adj[i] * (gen_adj[i]>=threshold).float().unsqueeze(dim=0)

            e1_recon_adj = lambda_one_list_adj(gen_adj)
            mean_value = np.mean(e1_recon_adj)
            threshold_means.append(mean_value)

        plt.plot(range(len(thresholds)), threshold_means, label="$alpha$" + alpha, color=color_palette[j])
        j += 1

    plt.xticks(range(len(thresholds)), thresholds)  # Set the tick locations and labels
    plt.xlabel("Threshold")
    plt.ylabel("Mean of lambda 1")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
              ncol=3, fancybox=True, shadow=True)
    plt.title("Mean of lambda 1 "+ dataset)
    plt.savefig(dataset + "_" + model + "_" + set + "_set_lambda_1_mean_over_thresholds.png", dpi=300)
    plt.show()
