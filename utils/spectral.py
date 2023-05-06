import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE


def first_pos_eigenvalue(a):
    """
    Returns the first positive eigenvalue of the laplacian of a graph
    :param a:
    :return:
    """
    return pos_eigenvalues(a)[1]


def pos_eigenvalues(a, device):
    """
    Returns the positive eigenvalues of the laplacian of a graph
    :param a:
    :return:
    """
    a.to(device)
    degrees = torch.sum(a, dim=1).unsqueeze(-1)
    I = torch.eye(a.size()[0])
    D = torch.pow(degrees, -0.5).squeeze()
    D = torch.diag(D)
    lap_sym = I - torch.mm(torch.mm(D, a), D)

    eigenvalues = torch.sort(torch.real(torch.linalg.eigvals(lap_sym)))
    return eigenvalues[0]


def add_laplacian_info_to_data(dataset):
    transform = AddLaplacianEigenvectorPE(5)
    data = []
    for graph in dataset:
        graph = transform(graph)
        data.append(graph)
    return data

def add_random_walk_info_to_data(dataset):
    transform = AddRandomWalkPE(5)
    data = []
    for graph in dataset:
        graph = transform(graph)
        data.append(graph)
    return data