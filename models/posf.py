# python module for position awareness functions
import torch
import math

# position awareness function selector
def position_weight(opt, x, aspect_double_idx, text_len, aspect_len):
    match opt.posf:
        case 'nill':
            return position_weight_nill(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_linear_mask':
            return position_weight_piecewise_linear_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_constant_mask':
            return position_weight_piecewise_constant_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_harmonic_mask':
            return position_weight_piecewise_harmonic_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_quadratic_mask':
            return position_weight_piecewise_quadratic_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_exponential_mask':
            return position_weight_piecewise_exponential_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_sigmoid_mask':
            return position_weight_piecewise_sigmoid_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_tanh_mask':
            return position_weight_piecewise_tanh_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_cosine_mask':
            return position_weight_piecewise_cosine_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_gaussian_mask':
            return position_weight_piecewise_gaussian_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case 'piecewise_sqrt_mask':
            return position_weight_piecewise_sqrt_mask(opt, x, aspect_double_idx, text_len, aspect_len)
        case _:
            print(f'Error: position awareness function "{opt.posf}" not found!')

# no position awareness
def position_weight_nill(opt, x, aspect_double_idx, text_len, aspect_len):
    return x

# position awareness using the piecewise linear mask function:
# 1-(j-i)/context_len if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_linear_mask(opt, x, aspect_double_idx, text_len, aspect_len):
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

# position awareness using the piecewise constant mask function:
# 1 if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_constant_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(1)
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(1)
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x

# position awareness using the piecewise harmonic mask function:
# 1/(j-i) if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_harmonic_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(1/(aspect_double_idx[i,0]-j))
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(1/(j-aspect_double_idx[i,1]))
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x

# position awareness using the piecewise quadratic mask function:
# 1-(j-i)^2/context_len^2 if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_quadratic_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        context_len = text_len[i] - aspect_len[i]
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(1-(aspect_double_idx[i,0]-j)**2/context_len**2)
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(1-(j-aspect_double_idx[i,1])**2/context_len**2)
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x

# position awareness using the piecewise exponential mask function:
# exp(-(j-i)/context_len) if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_exponential_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        context_len = text_len[i] - aspect_len[i]
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(math.exp(-(aspect_double_idx[i,0]-j)/context_len))
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(math.exp(-(j-aspect_double_idx[i,1])/context_len))
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x

# position awareness using the piecewise sigmoid mask function:
# 1/(1+exp(-(j-i)/context_len)) if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_sigmoid_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        context_len = text_len[i] - aspect_len[i]
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(1/(1+math.exp(-(aspect_double_idx[i,0]-j)/context_len)))
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(1/(1+math.exp(-(j-aspect_double_idx[i,1])/context_len)))
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x

# position awareness using the piecewise tanh mask function:
# 1/2*(1+tanh((j-i)/context_len)) if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_tanh_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        context_len = text_len[i] - aspect_len[i]
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(1/2*(1+math.tanh((aspect_double_idx[i,0]-j)/context_len)))
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(1/2*(1+math.tanh((j-aspect_double_idx[i,1])/context_len)))
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x

# position awareness using the piecewise cosine mask function:
# 1/2*(1+cos(pi*(j-i)/context_len)) if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_cosine_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        context_len = text_len[i] - aspect_len[i]
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(1/2*(1+math.cos(math.pi*(aspect_double_idx[i,0]-j)/context_len)))
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(1/2*(1+math.cos(math.pi*(j-aspect_double_idx[i,1])/context_len)))
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x

# position awareness using the piecewise gaussian mask function:
# exp(-(j-i)^2/context_len^2) if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_gaussian_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        context_len = text_len[i] - aspect_len[i]
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(math.exp(-(aspect_double_idx[i,0]-j)**2/context_len**2))
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(math.exp(-(j-aspect_double_idx[i,1])**2/context_len**2))
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x

# position awareness using the piecewise sqrt mask function:
# 1/sqrt(j-i) if j<i or j>i+aspect_len
# 0 if i<=j<=i+aspect_len
def position_weight_piecewise_sqrt_mask(opt, x, aspect_double_idx, text_len, aspect_len):
    batch_size = x.shape[0]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        context_len = text_len[i] - aspect_len[i]
        for j in range(aspect_double_idx[i,0]):
            weight[i].append(1/math.sqrt(aspect_double_idx[i,0]-j))
        for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, opt.max_seq_len)):
            weight[i].append(0)
        for j in range(min(aspect_double_idx[i,1]+1, opt.max_seq_len), text_len[i]):
            weight[i].append(1/math.sqrt(j-aspect_double_idx[i,1]))
    weight = torch.tensor(weight).unsqueeze(2).float().to(opt.device)
    return weight*x