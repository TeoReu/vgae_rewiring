import argparse
import torch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.nn import VGAE

from models.vgae import VariationalEncoder, L1VGAE, VariationalEncoderwithModel
from utils.dataset import split_dataset
from utils.plot import plot_paired_graphs, plot_paired_graph_weighted_edges
from utils.results import create_paths_vgae_weights, create_paths_vgae_experessivity_experiment
from utils.vgae import val, train, train_simple_vgae


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weights_path, model_outputs_path, model_pictures_path, file_path = create_paths_vgae_experessivity_experiment(args)

    train_set, test_set, val_set = split_dataset(args.transform)

    in_channels, out_channels, lr, n_epochs = train_set[0].num_features, 20, 0.001, 1

    gen_graphs, threshold, batch_size, add_self_loops = 3, 0.65, 20, False

    model = VGAE(VariationalEncoderwithModel(in_channels=in_channels, out_channels=out_channels, layers=args.layers, molecular=True, transform=args.transform, model=args.model))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    f = open(model_outputs_path, "w")
    best_acc = 0
    for epoch in range(1, n_epochs + 1):
        loss = train_simple_vgae(model, train_loader, optimizer, args, device)
        auc, ap = val(model, test_loader, args, device)
        if auc > best_acc:
            best_acc = auc
            best_model = model
        f.write(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}\n')
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')

    torch.save(best_model.state_dict(), model_weights_path)

    plot_paired_graph_weighted_edges(model, train_set, model_pictures_path, threshold, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--layers', type=int, default = 2)
    parser.add_argument('--file_name', type=str, default='new_vgae_cheeger')
    parser.add_argument('--transform', type=str, default="laplacian")
    parser.add_argument('--split_graph', type=str, default="_")

    args = parser.parse_args()
    main(args)
