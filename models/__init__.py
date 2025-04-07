from .dnn import CNN1D

def make(config, n_feats: int):


    if config.model == 'PS_AC1D_FIF':
        model = CNN1D.make(
            input_shape = n_feats,
            n_classes = len(config.class_labels),
            lr = config.lr
        )
    return model


_MODELS = {
    'PS_AC1D_FIF': CNN1D,
}

def load(config):
    return _MODELS[config.model].load(
        path = config.checkpoint_path,
        name = config.checkpoint_name
    )
