from utils.spectral import pos_eigenvalues, first_pos_eigenvalue


def lambda_one_adj(adj):
    return first_pos_eigenvalue(adj)


def lambda_one_list_adj(adj_list):
    return [lambda_one_adj(adj) for adj in adj_list]

