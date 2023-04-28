from typing import Optional, Union, Tuple

import torch
from torch_geometric.utils import to_dense_adj, negative_sampling
from torch_geometric.utils.num_nodes import maybe_num_nodes

def train(model, train_loader, optimizer, scheduler, args):
    model.train()
    loss_all = 0

    for data in train_loader:
        optimizer.zero_grad()
        z = model.encode(data)

        if args.split_graph == "_full_negative":
            loss = model.recon_loss(z, data.edge_index, negative_edges(data.edge_index)) + (
                        1 / data.num_nodes) * model.kl_loss() + args.alpha * model.lambda_loss(z)
        else:
            loss = model.recon_loss(z, data.edge_index, None) + (
                        1 / data.num_nodes) * model.kl_loss() + args.alpha * model.lambda_loss(z)

        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
        scheduler.step()

    return loss_all / len(train_loader.dataset)

@torch.no_grad()
def test(model, test_loader, gen_graphs):
    model.eval()
    gen_adj = []
    adj = []

    for graph, data in enumerate(test_loader):
        z = model.encode(data)
        gen_adj.append(model.decoder.forward_all(z))
        adj.append(to_dense_adj(data.edge_index))

        if graph == gen_graphs - 1:
            break
    return adj, gen_adj


@torch.no_grad()
def val(model, val_loader, args):
    model.eval()
    auc_all, ap_all = 0, 0

    for data in val_loader:
        z = model.encode(data)

        if args.split_graph == "_full_negative":
            auc, ap = model.test(z, data.edge_index, negative_edges(data.edge_index))
        else:
            auc, ap = model.test(z, data.edge_index, negative_sampling(data.edge_index, data.num_nodes))

        auc_all += data.y.size(0) * float(auc)
        ap_all += data.y.size(0) * float(ap)
    return auc_all / len(val_loader.dataset), ap_all / len(val_loader.dataset)


def negative_edges(edge_index: torch.Tensor,
                   num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                   force_undirected: bool = False) -> torch.Tensor:
    r"""Returns all negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        force_undirected (bool, optional): If set to :obj:`True`, returned
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor
    """
    size = num_nodes
    bipartite = isinstance(size, (tuple, list))
    size = maybe_num_nodes(edge_index) if size is None else size
    size = (size, size) if not bipartite else size
    force_undirected = False if bipartite else force_undirected

    complete_edges = torch.cartesian_prod(torch.arange(size[0]), torch.arange(size[1]))

    if force_undirected:
        complete_edges = complete_edges[complete_edges[:, 0] < complete_edges[:, 1]]

    # create a mask of the same shape as complete_edges
    mask = torch.zeros(complete_edges.shape[0], dtype=torch.bool)

    # mark the positions of edges in edge_index as True
    mask[edge_index.T[:, 0] * complete_edges.shape[1] + edge_index.T[:, 1]] = True

    # select negative edges using the mask
    neg_edges = complete_edges[~mask]
    return neg_edges.T