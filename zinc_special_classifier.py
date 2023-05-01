import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

from models.special_supervised import VGAEGCN_init_vae
from models.supervized import GCN, TGCN
from models.vgae import VariationalEncoder, L1VGAE
from utils.dataset import split_dataset
from utils.results import create_paths_for_classifier



def main(args):
    np.random.seed(args.nr)
    torch.manual_seed(args.nr)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if args.transform == 1:
        args.transform = True
    else:
        args.transform = False

    model_outputs_path, file_path = create_paths_for_classifier(args)

    data_train, data_test, data_val, = split_dataset(args.transform)

    in_channels, out_channels, lr, n_epochs = data_train[0].num_features, 20, 0.001, 100


    vae_layers, alpha, threshold = args.vae_layers, args.alpha, args.threshold
    vae = L1VGAE(VariationalEncoder(in_channels, out_channels, layers=vae_layers, molecular=True, transform=args.transform), device)
    vae.load_state_dict(torch.load(
        'results/model_GCN/graph_split_/layers_' + str(
            vae_layers) + '/transform_'+str(args.transform)+'/alpha_' + str(
            alpha) +'/model.pt'))


    model = VGAEGCN_init_vae(input_dim=data_train[0].num_features, output_dim=1, hidden_dim=32, vae=vae, num_layers=args.layers, molecular=True, trans=args.transform)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Split the dataset into train and test sets

    train_loader = DataLoader(data_train, batch_size=20, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=20)

    # Train the model for 100 epochs

    f = open(model_outputs_path, "w")

    best_acc = 0
    for epoch in range(1, n_epochs):
        train(model, train_loader, optimizer)
        acc = v_test(model, test_loader)
        if acc[1] > best_acc:
            best_acc = acc[1]
            torch.save(vae.state_dict(), file_path+'/model.pt')

        print(f'Epoch: {epoch:03d}, MSE: {acc[0]:.4f}, L1: {acc[1]:.4f}')
        f.write(f'Epoch: {epoch:03d}, MSE: {acc[0]:.4f}, L1: {acc[1]:.4f}\n')
    f.close()

def transform_zinc_dataset(vae, dataset, threshold):
    dataset_copy = []
    for graph in dataset:
        #print(graph)
        z = vae.encode(graph)
        gen_adj = vae.decoder.forward_all(z) > threshold
        sparse, attr = dense_to_sparse(gen_adj)
        graph.vr_edge_index = sparse
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
    parser.add_argument('--model', type=str, default="simple")
    parser.add_argument('--vae_layers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--file_name', type=str, default="test/zinc_models")
    parser.add_argument('--transform', type=str, default=True)
    parser.add_argument('--nr', type=int, default=1)
    args = parser.parse_args()
    main(args)
