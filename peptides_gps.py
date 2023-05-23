import torch
import argparse
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
import torch_geometric.transforms as T
from utils.peptides_dataset import PeptidesStructuralDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from sklearn.metrics import average_precision_score
import numpy as np

class GPS(torch.nn.Module):
    def __init__(self, channels: int, num_layers: int, edge_conv: str):
        super().__init__()

        self.node_emb = Linear(9, channels)
        self.pe_lin = Linear(20, channels)
        self.edge_emb = Linear(3, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if edge_conv == "GINE":
              conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5)
            else:
              conv = GPSConv(channels, None, heads=4, attn_dropout=0.5)

            self.convs.append(conv)

        self.lin = Linear(channels, 11)

    def forward(self, data):
        pe = self.pe_lin(data.pe)
        x = self.node_emb(data.x.float())
        x = x + pe
        edge_attr = self.edge_emb(data.edge_attr.float())

        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch, edge_attr=edge_attr)
        x = global_add_pool(x, data.batch)
        return self.lin(x)


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
        total_loss.append(loss.detach()) #* data.num_graphs
        optimizer.step()
    return torch.mean(torch.stack(total_loss))


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


def main(args):
  path = "transformers/peptides/" + args.conv + "_conv/"
  dataset_1 = PeptidesStructuralDataset()
  transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

  dataset = []
  for graph in dataset_1:
      graph = transform(graph)
      dataset.append(graph)


      # Create training, validation, and test sets
  train_dataset = dataset[:int(len(dataset) * 0.8)]
  val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
  test_dataset = dataset[int(len(dataset) * 0.9):]

  print(f'Training set   = {len(train_dataset)} graphs')
  print(f'Validation set = {len(val_dataset)} graphs')
  print(f'Test set       = {len(test_dataset)} graphs')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = GPS(channels=64, num_layers=args.layers, edge_conv=args.conv).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  val_mae_min = 1000.0
  best_model = None
  torch.save(model.state_dict(), path+'gps_first.pt')
  f = open(path + "results.txt", "w")

  for epoch in range(1, 301):
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


  torch.save(best_model.state_dict(), path+'gps_best.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='peptides')
    parser.add_argument('--conv', type=str, default='GINE')
    parser.add_argument('--layers', type=int, default = 5)

    args = parser.parse_args()
    main(args)


