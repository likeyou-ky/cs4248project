import torch.nn.functional as F

def actf(name):
    match name:
        case 'relu':
            return F.relu
        case 'tanh':
            return F.tanh
        case 'sigmoid':
            return F.sigmoid
        case 'softmax':
            return F.softmax
        case 'log_softmax':
            return F.log_softmax
        case 'leaky_relu':
            return F.leaky_relu
        case 'elu':
            return F.elu
        case 'selu':
            return F.selu
        case 'gelu':
            return F.gelu
        case 'softsign':
            return F.softsign
        case 'mish':
            return F.mish
        case 'hardshrink':
            return F.hardshrink
        case 'tanhshrink':
            return F.tanhshrink
        case 'threshold':
            return F.threshold
        case 'hardtanh':
            return F.hardtanh
        case 'softmin':
            return F.softmin
        case 'none':
            return lambda x: x
        case _:
            raise NotImplementedError('Activation function has not been implemented: {}'.format(name))