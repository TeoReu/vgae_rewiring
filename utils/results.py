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

def create_paths_vgae_experessivity_experiment(args):
    file_path = args.file_name + "/model_" + args.model + "/layers_" + str(
        args.layers) + "/transform_" + str(args.transform)

    model_weights_path = file_path + "/model.pt"
    model_outputs_path = file_path + "/results.txt"
    model_pictures_path = file_path + "/graphs"

    if not os.path.exists(model_pictures_path):
        os.makedirs(model_pictures_path)

    return model_weights_path, model_outputs_path, model_pictures_path, file_path


def create_paths_for_classifier(args):
    if args.model == 'simple':
        file_path = args.file_name + "/model_" + args.model + "/layers_" + str(args.layers)  + '/nr_' + str(args.nr) + '/transform_' + str(args.transform)
    else:
        file_path = args.file_name + "/model_" + args.model + "/layers_" + str(
            args.layers) + "/vae_layers_" + str(args.vae_layers) + "/threshold_" + str(args.threshold) + '/transform_ ' + str(args.transform) + '/alpha_' + str(args.alpha) + '/nr_' + str(args.nr)
    model_outputs_path = file_path + "/results.txt"

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    return model_outputs_path, file_path