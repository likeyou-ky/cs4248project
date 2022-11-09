import torch.nn.functional as F

def actf(name):
    try:
        return getattr(F, name)
    except AttributeError:
        print('Activation function has not been implemented: {}'.format(name))

    match name:
        case 'relu':
            return F.relu
        case 'leaky_relu':
            return F.leaky_relu
        case 'tanh':
            return F.tanh
        case 'sigmoid':
            return F.sigmoid
        case 'softmax':
            return F.softmax
        case 'log_softmax':
            return F.log_softmax
        case 'softsign':
            return F.softsign
        case 'elu':
            return F.elu
        case 'selu':
            return F.selu
        case 'celu':
            return F.celu
        case 'glu':
            return F.glu
        case 'hardshrink':
            return F.hardshrink
        case 'hardtanh':
            return F.hardtanh
        case 'hardsigmoid':
            return F.hardsigmoid
        case 'hardswish':
            return F.hardswish
        case 'softmin':
            return F.softmin
        case 'tanhshrink':
            return F.tanhshrink
        case 'threshold':
            return F.threshold
        case _:
            raise NotImplementedError('Activation function has not been implemented: {}'.format(name))