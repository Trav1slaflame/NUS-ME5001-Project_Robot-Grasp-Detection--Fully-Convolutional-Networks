def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    #Add Residual Block to original GGCNN
    elif network_name == 'grcnn':
        from .grcnn import GRCNN
        return GRCNN
    elif network_name == 'grcnn2':
        from .grcnn2 import GRCNN2
        return GRCNN2
    elif network_name == 'grcnn3':
        from .grcnn3 import GRCNN3
        return GRCNN3
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
