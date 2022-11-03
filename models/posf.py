# python module for position awareness functions
import torch

# position awareness function selector
def position_weight(opt, x, aspect_double_idx, text_len, aspect_len):
    if opt.posf == 'nill':
        return position_weight_nill(opt, x, aspect_double_idx, text_len, aspect_len)
    elif opt.posf == 'piecewise_mask':
        return position_weight_piecewise_mask(opt, x, aspect_double_idx, text_len, aspect_len)
    else:
        raise ValueError('Unknown position awareness function: ' + opt.posf)

# no position awareness
def position_weight_nill(opt, x, aspect_double_idx, text_len, aspect_len):
    return x

# position awareness using the piecewise mask function:
# 1-(j-i)/context_len if i<=j<=i+context_len
# 0 if j<i or j>i+context_len
def position_weight_piecewise_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    #batch_size = len(x)
    #seq_len = len(x[1])
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    text_len = text_len.cpu().numpy()
    aspect_len = aspect_len.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        context_len = text_len[i] - aspect_len[i]
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1,opt.max_seq_len)):
            weight[i].append(0)
        for j in range(aspect_double_idx[i,1]+1, text_len[i]):
            weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
        for j in range(text_len[i], seq_len):
            weight[i].append(0)
    weight = torch.tensor(weight).unsqueeze(2).to(opt.device)
    return weight*x