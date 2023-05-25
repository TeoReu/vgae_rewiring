import os

import numpy
import torch
import argparse
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.utils import dense_to_sparse

from models.vgae import L1VGAE, VariationalEncoderwithModel
from utils.peptides_dataset import PeptidesStructuralDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, GCNConv
from sklearn.metrics import average_precision_score
import numpy as np
import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

from utils.results import create_paths_vgae_experessivity_experiment
from torch_geometric.utils import degree
import os.path as osp


class GPSConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        conv1: Optional[MessagePassing],
        conv2: Optional[MessagePassing],
        dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv1 = conv1
        self.conv2 = conv2
        self.dropout = dropout

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv1 is not None:
            self.conv1.reset_parameters()
        if self.conv2 is not None:
            self.conv2.reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        vae_edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        vae_edge_weight: Optional[Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv1 is not None:  # Local MPNN.
            h = self.conv1(x, edge_index, edge_attr=edge_attr, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        if self.conv2 is not None:  # Local MPNN.
            h = self.conv2(x, vae_edge_index, edge_weight=vae_edge_weight, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')


class GPS(torch.nn.Module):
    def __init__(self, channels: int, num_layers: int, args):
        super().__init__()

        self.node_emb = Linear(9, channels)
        self.pe_lin = Linear(args.pe, channels)
        self.edge_emb = Linear(3, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )

            if args.conv == "GINE":
                conv = GPSConv(channels, GINEConv(nn), GCNConv(in_channels = channels, out_channels =channels), dropout= 0.2)
            else:
                conv = GPSConv(channels, None, GCNConv(in_channels = channels, out_channels =channels), dropout= 0.2)

            self.convs.append(conv)

        self.lin = Linear(channels, 11)

    def forward(self, data):
        pe = self.pe_lin(data.pe)
        x = self.node_emb(data.x.float())
        x = x + pe
        edge_attr = self.edge_emb(data.edge_attr.float())

        for conv in self.convs:
            x = conv(x, data.edge_index, data.vae_edge_index, edge_attr, data.vae_edge_weight, data.batch)
        x = global_add_pool(x, data.batch)
        return self.lin(x)

def create_paths(args):
    file_path =  args.file_name +"/conv" + str(args.conv) + "/pe" + str(args.pe) +  "/threshold" + str(args.threshold) + "/alpha_" + str(args.alpha)

    model_weights_path = file_path + "/model.pt"
    model_outputs_path = file_path + "/results.txt"

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    return model_weights_path, model_outputs_path, file_path



def train(epoch, model, optimizer, train_loader, device):
    model.train()
    criterion = torch.nn.functional.l1_loss
    total_loss = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.cpu().detach().numpy())

    return numpy.mean(numpy.stack(total_loss))


@torch.no_grad()
def test(loader, model, device):
    model.eval()
    criterion = torch.nn.functional.l1_loss
    total_error = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        total_error.append(criterion(out.squeeze(), data.y))
    return torch.mean(torch.stack(total_error))

@torch.no_grad()
def transform_dataset_with_weights(vae, dataset, threshold):
    vae.eval()
    dataset_copy = []
    for graph in dataset:
        #print(graph)
        z = vae.encode(graph)
        gen_adj = vae.decoder.forward_all(z)

        gen_adj = ReLU()(gen_adj - threshold)

        new_edge_index, new_edge_weight = dense_to_sparse(gen_adj)

        graph.vae_edge_index = new_edge_index
        graph.vae_edge_weight = new_edge_weight
        dataset_copy.append(graph)
    return dataset_copy

def retrive_vae(alpha, train_set):
    max_degree = -1
    for data in train_set:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_set:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = L1VGAE(VariationalEncoderwithModel(in_channels=train_set[0].num_features, out_channels=200, layers=4, molecular=False, transform="no", model="PNA", deg=deg, edge_dim=1), device)

    vae.load_state_dict(torch.load("vgae/peptides/model_PNA/layers_4/transform_no/alpha_"+ str(alpha) +"/model.pt",  map_location=device))
    return vae

def main(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC-PE')

    model_weights_path, model_outputs_path, file_path = create_paths(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.AddRandomWalkPE(walk_length=args.pe, attr_name='pe')
    train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
    val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
    test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)

    vae = retrive_vae(alpha=args.alpha)

    train_dataset = transform_dataset_with_weights(vae, train_dataset, args.threshold)
    val_dataset = transform_dataset_with_weights(vae, val_dataset, args.threshold)
    test_dataset = transform_dataset_with_weights(vae, test_dataset, args.threshold)


    print(f'Training set   = {len(train_dataset)} graphs')
    print(f'Validation set = {len(val_dataset)} graphs')
    print(f'Test set       = {len(test_dataset)} graphs')

    model = GPS(channels=64, num_layers=args.layers, args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    val_mae_min = 1000.0
    best_model = None
    f = open(model_outputs_path, "w")

    for epoch in range(1, 151):
          optimizer.zero_grad()
          loss = train(epoch, model, optimizer, train_loader, device)
          val_mae = test(val_loader, model, device)

          if  val_mae < val_mae_min:
              val_mae_min = val_mae
              best_model = model

          test_mae = test(test_loader, model, device)
          f.write(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
                f'Test: {test_mae:.4f}\n')

          print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
                f'Test: {test_mae:.4f}')


    torch.save(best_model.state_dict(), model_weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='peptides')
    parser.add_argument('--conv', type=str, default='GINE')
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--file_name', type=str, default='zinc_results')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--pe', type=int, default=2)

    args = parser.parse_args()
    main(args)


