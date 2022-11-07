# python module for mask functions
import torch

# mask function selector
def mask(opt, x, aspect_double_idx):
    if opt.mask == 'nill':
        return x
    f = globals()[opt.mask]
    return mask_general(opt, x, aspect_double_idx, f)

# general function for mask
def mask_general(opt, x, aspect_double_idx, func):
    batch_size, seq_len = x.shape[0], x.shape[1]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    mask = [[] for i in range(batch_size)]
    for i in range(batch_size):
        aspect_start = aspect_double_idx[i,0]
        aspect_end = aspect_double_idx[i,1]
        func(aspect_start, aspect_end, seq_len, mask[i])
    mask = torch.tensor(mask).unsqueeze(2).float().to(opt.device)
    return mask*x

# uniform aspect mask
def uniform_aspect_mask(aspect_start, aspect_end, seq_len, mask):
    for j in range(aspect_start):
        mask.append(0)
    for j in range(aspect_start, aspect_end+1):
        mask.append(1)
    for j in range(aspect_end+1, seq_len):
        mask.append(0)