from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from utils.peptides_dataset import PeptidesFunctionalDataset
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

def split_dataset_peptides(transform=None):
    dataset = PeptidesFunctionalDataset()
    print(dataset[100])

    # Create training, validation, and test sets
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    if transform == "laplacian":
        train_dataset = add_laplacian_info_to_data(train_dataset)
        val_dataset = add_laplacian_info_to_data(val_dataset)
        test_dataset = add_laplacian_info_to_data(test_dataset)
    elif transform == "random_walk":
        train_dataset = add_random_walk_info_to_data(train_dataset)
        val_dataset = add_random_walk_info_to_data(val_dataset)
        test_dataset = add_random_walk_info_to_data(test_dataset)
    else:
        pass

    return train_dataset, test_dataset, val_dataset