import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

from models.supervized import GCN, TGCN
from models.vgae import VariationalEncoder, L1VGAE, VariationalEncoderwithModel
from utils.dataset import split_dataset, split_dataset_peptides
from utils.results import create_paths_for_classifier
from torch_geometric.utils import degree



def main(args):
    np.random.seed(args.nr)
    torch.manual_seed(args.nr)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "ZINC":
        train_set, test_set, val_set = split_dataset(args.transform)
        edge_attr_dim = 1
        batch_size = 20
    else:
        train_set, test_set, val_set = split_dataset_peptides(args.transform)
        edge_attr_dim = train_set[0].edge_attr.shape[1]
        batch_size = 3

    deg = None
    if args.model == "PNA":
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


    vae_layers, alpha, threshold = args.vae_layers, args.alpha, args.threshold
    vae = L1VGAE(VariationalEncoderwithModel(in_channels=in_channels, out_channels=out_channels, layers=args.layers, molecular=False, transform=args.transform, model=args.model, deg=deg, edge_dim=edge_attr_dim), device)

    vae.load_state_dict(torch.load(
        'vgae/model_PNA/layers_4/transform_no'+'/alpha_' + str(
            alpha) +'/model.pt'))

    vae.eval()

    data_train = transform_zinc_dataset(vae, data_train, args.threshold)
    data_test = transform_zinc_dataset(vae, data_test, args.threshold)
    model = TGCN(data_train[0].num_features, 1, 32, args.layers, molecular=True, trans=args.transform, vae=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Split the dataset into train and test sets

    train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=64)

    # Train the model for 100 epochs

    f = open(model_outputs_path, "w")

    for epoch in range(1, n_epochs):
        train(model, train_loader, optimizer)
        acc = v_test(model, test_loader)
        print(f'Epoch: {epoch:03d}, MSE: {acc[0]:.4f}, L1: {acc[1]:.4f}')
        f.write(f'Epoch: {epoch:03d}, MSE: {acc[0]:.4f}, L1: {acc[1]:.4f}\n')
    f.close()

def transform_zinc_dataset(vae, dataset, threshold):
    dataset_copy = []
    for graph in dataset:
        #print(graph)
        z = vae.encode(graph)
        gen_adj = vae.decoder.forward_all(z) * (vae.decoder.forward_all(z)>threshold).float()
        sparse, attr = dense_to_sparse(gen_adj)
        graph.vr_edge_index = sparse
        graph.vr_edge_weight = attr
        dataset_copy.append(graph)

    return dataset_copy

def transform_zinc_dataset_with_weights(vae, dataset, threshold):
    dataset_copy = []
    for graph in dataset:
        #print(graph)
        z = vae.encode(graph)
        gen_adj = vae.decoder.forward_all(z)

        gen_adj = ReLU()(gen_adj - threshold)

        new_edge_index, new_edge_weight = dense_to_sparse(gen_adj)

        graph.vr_edge_index = new_edge_index
        graph.vr_edge_weight = new_edge_weight
        dataset_copy.append(graph)
    return dataset_copy
# Define the training loop
def train(model, train_loader, optimizer):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out, _ = model(data)
        loss = F.l1_loss(out, data.y)
        loss.backward()
        optimizer.step()

# Define the test loop
def v_test(model, loader):
    model.eval()
    l1_loss = 0.0
    mse_loss = 0.0
    for data in loader:
        out, _ = model(data)
        mse_loss += F.mse_loss(out, data.y)
        l1_loss += F.l1_loss(out, data.y)
    return (mse_loss / len(loader), l1_loss / len(loader))

# Initialize the model and the optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="vr")
    parser.add_argument('--vae_layers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--file_name', type=str, default="test/zinc_models")
    parser.add_argument('--transform', type=int, default=1)
    parser.add_argument('--nr', type=int, default=1)
    args = parser.parse_args()
    main(args)
