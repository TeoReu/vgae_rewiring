from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import GCN, VGAE
from torch_scatter import scatter_sum

from models.vgae import VariationalEncoder


class VGAEGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2,
                 molecular=True, trans=True, vae=True):
        super(GCN, self).__init__()
        self.num_layers = num_layers  # please select num_layers>=2
        self.molecular = molecular
        self.trans = trans
        self.vae = vae

        if self.molecular:
            self.embed_x = Embedding(28, hidden_dim)
        else:
            self.embed_x = Linear(input_dim, hidden_dim)

        self.vgae = VGAE(VariationalEncoder(in_channels=1, out_channels=hidden_dim, layers=2, molecular=True, transform=False))

        if self.trans:
            self.lin_trans = Linear(5, hidden_dim)

        self.layers = GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=output_dim)


    def forward(self, data):
        z = self.vgae.encode(data)

        gen_adj = self.vgae.decoder.forward_all(z)

        new_edge_index = to_


        x_1 = self.layers(x, data.edge_index)

        if self.vae:
            x_vae = self.vae_layers(x, data.vr_edge_index)
            x_1 = x_vae + x_1

        y_hat = scatter_sum(x_1, data.batch, dim=0)
        y_hat = y_hat.squeeze(-1)
        return y_hat, x
