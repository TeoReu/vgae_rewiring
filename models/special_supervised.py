from torch import nn
from torch.nn import Embedding, Linear, ReLU
from torch_geometric.nn import GCN, VGAE
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter_sum

from models.vgae import VariationalEncoder


class VGAEGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2,
                 molecular=True, trans=False):
        super(VGAEGCN, self).__init__()
        self.num_layers = num_layers  # please select num_layers>=2
        self.molecular = molecular
        self.trans = trans

        if self.molecular:
            self.embed_x = Embedding(28, hidden_dim)
        else:
            self.embed_x = Linear(input_dim, hidden_dim)

        self.vgae = VGAE(VariationalEncoder(in_channels=1, out_channels=hidden_dim, layers=2, molecular=True, transform=False))

        self.vgae_layers = GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=output_dim)
        self.layers= GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=output_dim)

    def forward(self, data):
        x = self.embed_x(data.x.long()).squeeze(1)


        z = self.vgae.encode(data)

        gen_adj = self.vgae.decoder.forward_all(z)
        gen_adj = ReLU()(gen_adj - 0.65)

        new_edge_index, new_edge_weight = dense_to_sparse(gen_adj)

        x_0 = self.layers(x, data.edge_index)
        x_1 = self.vgae_layers(x, new_edge_index, edge_weight=new_edge_weight)

        x = x_0 + x_1
        y_hat = scatter_sum(x, data.batch, dim=0)
        y_hat = y_hat.squeeze(-1)
        return y_hat, data.x


class VGAEGCN_init_vae(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, vae, threshold =0.65, num_layers=2,
                 molecular=True, trans=False):
        super(VGAEGCN_init_vae, self).__init__()
        self.num_layers = num_layers  # please select num_layers>=2
        self.molecular = molecular
        self.trans = trans
        self.threshold = threshold
        if self.molecular:
            self.embed_x = Embedding(28, hidden_dim)
        else:
            self.embed_x = Linear(input_dim, hidden_dim)

        self.vgae = vae

        self.vgae_layers = GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=output_dim)
        self.layers= GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=output_dim)

    def forward(self, data):
        x = self.embed_x(data.x.long()).squeeze(1)


        z = self.vgae.encode(data)

        gen_adj = self.vgae.decoder.forward_all(z)
        gen_adj = ReLU()(gen_adj - self.threshold)

        new_edge_index, new_edge_weight = dense_to_sparse(gen_adj)

        x_0 = self.layers(x, data.edge_index)
        x_1 = self.vgae_layers(x, new_edge_index, edge_weight=new_edge_weight)

        x = x_0 + x_1
        y_hat = scatter_sum(x, data.batch, dim=0)
        y_hat = y_hat.squeeze(-1)
        return y_hat, data.x


class NormalGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2,
                 molecular=True, trans=False):
        super(NormalGCN, self).__init__()
        self.num_layers = num_layers  # please select num_layers>=2
        self.molecular = molecular
        self.trans = trans

        if self.molecular:
            self.embed_x = Embedding(28, hidden_dim)
        else:
            self.embed_x = Linear(input_dim, hidden_dim)

        self.vgae = VGAE(VariationalEncoder(in_channels=1, out_channels=hidden_dim, layers=2, molecular=True, transform=False))

        self.vgae_layers = GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=output_dim)
        self.layers= GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=output_dim)

    def forward(self, data):
        x = self.embed_x(data.x.long()).squeeze(1)

        x = self.layers(x, data.edge_index)

        y_hat = scatter_sum(x, data.batch, dim=0)
        y_hat = y_hat.squeeze(-1)
        return y_hat, data.x