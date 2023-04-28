import os


def create_paths_vgae_weights(args):
    file_path = args.file_name + "/model_" + args.model + "/graph_split" + str(args.split_graph) + "/layers_" + str(
        args.layers) + "/transform_" + str(args.transform) + "/alpha_" + str(args.alpha)

    model_weights_path = file_path + "/model.pt"
    model_outputs_path = file_path + "/results.txt"
    model_pictures_path = file_path + "/graphs"

    if not os.path.exists(model_pictures_path):
        os.makedirs(model_pictures_path)

    return model_weights_path, model_outputs_path, model_pictures_path, file_path
