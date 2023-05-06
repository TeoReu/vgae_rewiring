from torch_geometric.datasets import ZINC

from utils.spectral import add_laplacian_info_to_data, add_random_walk_info_to_data


def split_dataset(transform=None):
    data_train = ZINC(root='/tmp/ZINC', split='train', subset=True)
    data_test = ZINC(root='/tmp/ZINC', split='test', subset=True)

    if transform == "laplacian":
        data_train = add_laplacian_info_to_data(data_train)
        data_test = add_laplacian_info_to_data(data_test)
    elif transform == "random_walk":
        data_train = add_random_walk_info_to_data(data_train)
        data_test = add_random_walk_info_to_data(data_test)
    else:
        pass

    data_val = data_train[9500:10000]
    return data_train, data_test, data_val