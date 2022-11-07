import torch.nn.functional as F

def actf(name):
    try:
        return getattr(F, name)
    except AttributeError:
        print('Activation function has not been implemented: {}'.format(name))