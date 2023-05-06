import torch
from torch.nn import Embedding, Linear, ModuleList, Sequential, ReLU
from torch.nn import Module
from typing import Optional
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GCN, VGAE, GIN, GINConv, PNA, PNAConv, BatchNorm
from utils.spectral import first_pos_eigenvalue

class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers, molecular=True, transform=True):
        super().__init__()

        self.molecular = molecular
        self.transform = transform

        '''
        if type == "GIN" and layers > 1:
            self.conv = GIN(in_channels=2 * out_channels, hidden_channels=2 * out_channels, num_layers=layers,
                             out_channels=out_channels)
        elif type == "GCN" and layers > 1:
            self.conv = GCN(in_channels=2 * out_channels, hidden_channels=2 * out_channels, num_layers=layers,
                             out_channels=out_channels)
        elif type == "GIN" and layers == 1:
            self.conv = GINConv(in_channels=2 * out_channels, out_channels=out_channels)
        else:
            self.conv = GCNConv(in_channels=2 * out_channels, out_channels=out_channels)
        '''
        if layers == 1:
            self.conv = GCNConv(in_channels=2 * out_channels, out_channels=out_channels)
        else:
            self.conv = GCN(in_channels=2 * out_channels, hidden_channels=2 * out_channels, num_layers=layers,
                            out_channels=out_channels)

        if self.molecular:
            self.embed_x = Embedding(28, 2 * out_channels)
        else:
            self.embed_x = Linear(in_channels, 2 * out_channels)


        if self.transform:
            self.trans_linear = Linear(5, 2 * out_channels)

        self.conv_mu = GCNConv(out_channels, out_channels)
        self.conv_logstd = GCNConv(out_channels, out_channels)

    def forward(self, graph):
        if self.molecular:
            x = self.embed_x(graph.x.long()).squeeze(1)
        else:
            x = self.embed_x(graph.x)

        if self.transform:
            x_pe = self.trans_linear(graph.laplacian_eigenvector_pe)
            x = x + x_pe

        x = self.conv(x, graph.edge_index).relu()
        return self.conv_mu(x, graph.edge_index), self.conv_logstd(x, graph.edge_index)



class VariationalEncoderwithModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers, molecular=True, transform=True, model="GCN", deg=3):
        super().__init__()

        self.molecular = molecular
        self.transform = transform
        self.layers = layers
        self.model = model

        if self.model == "GIN" and self.layers > 1:
            self.conv = GIN(in_channels=2 * out_channels, hidden_channels=2 * out_channels, num_layers=self.layers,
                             out_channels=out_channels)
        elif self.model == "GCN" and self.layers > 1:
            self.conv = GCN(in_channels=2 * out_channels, hidden_channels=2 * out_channels, num_layers=self.layers,
                             out_channels=out_channels)
        elif self.model == "PNA":
            self.conv = PNANet(layers=self.layers, out_channels=out_channels, deg=deg)
        elif self.model == "GIN" and self.layers == 1:
            self.conv = GINConv(in_channels=2 * out_channels, out_channels=out_channels)
        elif self.model == "GCN" and self.layers == 1:
            self.conv = GCNConv(in_channels=2 * out_channels, out_channels=out_channels)


        if self.molecular:
            self.embed_x = Embedding(28, 2 * out_channels)
        else:
            self.embed_x = Linear(in_channels, 2 * out_channels)


        if self.transform:
            self.trans_linear = Linear(5, 2 * out_channels)

        self.conv_mu = GCNConv(out_channels, out_channels)
        self.conv_logstd = GCNConv(out_channels, out_channels)

    def forward(self, graph):
        if self.molecular:
            x = self.embed_x(graph.x.long()).squeeze(1)
        else:
            x = self.embed_x(graph.x)

        if self.transform == "laplacian":
            x_pe = self.trans_linear(graph.laplacian_eigenvector_pe)
            x = x + x_pe
        elif self.transform == "random_walk":
            x_pe = self.trans_linear(graph.random_walk_pe)
            x = x + x_pe
        else:
            pass

        if self.model == "PNA":
            x = self.conv(x, graph.edge_index, graph.edge_attr).relu()
        else:
            x = self.conv(x, graph.edge_index).relu()
        return self.conv_mu(x, graph.edge_index), self.conv_logstd(x, graph.edge_index)

class L1VGAE(VGAE):
    def __init__(self, encoder: Module, device, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)

        self.device = device

    def lambda_loss(self, z):
        a = self.decoder.forward_all(z)

        degrees = torch.sum(a, dim=1).unsqueeze(-1)
        I = torch.eye(a.size()[0])

        D = torch.pow(degrees, -0.5).squeeze()
        D = torch.diag(D)

        lap_sym = I.to(self.device) - torch.mm(torch.mm(D, a).to(self.device), D.to(self.device))

        eigenvalues = torch.sort(torch.real(torch.linalg.eigvals(lap_sym)))

        return - eigenvalues[0][1]


class PNANet(torch.nn.Module):
    def __init__(self, layers, out_channels, deg):
        super().__init__()

        self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(layers):
            conv = PNAConv(in_channels=2 * out_channels, out_channels=2 * out_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(2 * out_channels))

        self.mlp = Sequential(Linear(2 * out_channels, 2 * out_channels), ReLU(), Linear(2 * out_channels,  out_channels))

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        return self.mlp(x)