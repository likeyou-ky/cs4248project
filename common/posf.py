# python module for position awareness functions
import torch
import math

# position awareness function selector
def position_weight(opt, x, aspect_double_idx, text_len, aspect_len):
    if opt.posf == 'nill':
        return x
    f = globals()[opt.posf]
    return position_weight_general(opt, x, aspect_double_idx, text_len, aspect_len, f)

# general function for position awareness
def position_weight_general(opt, x, aspect_double_idx, text_len, aspect_len, func):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    aspect_double_idx = aspect_double_idx.cpu().numpy()
    text_len = text_len.cpu().numpy()
    aspect_len = aspect_len.cpu().numpy()
    weight = [[] for i in range(batch_size)]
    for i in range(batch_size):
        aspect_start = aspect_double_idx[i,0]
        aspect_end = aspect_double_idx[i,1]
        context_len = text_len[i] - aspect_len[i]
        func(aspect_start, aspect_end, context_len, seq_len, opt.max_seq_len, text_len[i], weight[i])
    weight = torch.tensor(weight).unsqueeze(2).to(opt.device)
    return weight*x

# position awareness using the piecewise linear mask function:
# 1-(aspect_start-j)/context_len    if j<aspect_start
# 0                                 if aspect_start<=j<=aspect_end
# 1-(j-i)/context_len               if aspect_end<j<text_len
# 0                                 if text_len<=j<seq_len
def piecewise_linear_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(1-(aspect_start-j)/context_len)
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(1-(j-aspect_end)/context_len)
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise constant mask function:
# 1 if j<aspect_start
# 0 if aspect_start<=j<=aspect_end
# 1 if aspect_end<j<text_len
# 0 if text_len<=j<seq_len
def piecewise_constant_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(1)
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(1)
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise harmonic mask function:
# 1/(aspect_start-j)    if j<aspect_start
# 0                     if aspect_start<=j<=aspect_end
# 1/(j-aspect_end)      if aspect_end<j<text_len
# 0                     if text_len<=j<seq_len
def piecewise_harmonic_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(1/(aspect_start-j))
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(1/(j-aspect_end))
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise quadratic mask function:
# 1-(aspect_start-j)^2/context_len^2    if j<aspect_start
# 0                                     if aspect_start<=j<=aspect_end
# 1-(j-aspect_end)^2/context_len^2      if aspect_end<j<text_len
# 0                                     if text_len<=j<seq_len
def piecewise_quadratic_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(1-(aspect_start-j)**2/context_len**2)
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(1-(j-aspect_end)**2/context_len**2)
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise exponential mask function:
# exp(-(aspect_start-j)/context_len)    if j<aspect_start
# 0                                     if aspect_start<=j<=aspect_end
# exp(-(j-aspect_end)/context_len)      if aspect_end<j<text_len
# 0                                     if text_len<=j<seq_len
def piecewise_exponential_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(math.exp(-(aspect_start-j)/context_len))
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(math.exp(-(j-aspect_end)/context_len))
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise sigmoid mask function:
# 1/(1+exp(-(aspect_start-j)/context_len))  if j<aspect_start
# 0                                         if aspect_start<=j<=aspect_end
# 1/(1+exp(-(j-aspect_end)/context_len))    if aspect_end<j<text_len
# 0                                         if text_len<=j<seq_len
def piecewise_sigmoid_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(1/(1+math.exp(-(aspect_start-j)/context_len)))
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(1/(1+math.exp(-(j-aspect_end)/context_len)))
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise tanh mask function:
# 1/2*(1+tanh((aspect_start-j)/context_len))    if j<aspect_start
# 0                                             if aspect_start<=j<=aspect_end
# 1/2*(1+tanh((j-aspect_end)/context_len))      if aspect_end<j<text_len
# 0                                             if text_len<=j<seq_len
def piecewise_tanh_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(1/2*(1+math.tanh((aspect_start-j)/context_len)))
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(1/2*(1+math.tanh((j-aspect_end)/context_len)))
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise cosine mask function:
# 1/2*(1+cos((aspect_start-j)/context_len)) if j<aspect_start
# 0                                         if aspect_start<=j<=aspect_end
# 1/2*(1+cos((j-aspect_end)/context_len))   if aspect_end<j<text_len
# 0                                         if text_len<=j<seq_len
def piecewise_cosine_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(1/2*(1+math.cos((aspect_start-j)/context_len)))
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(1/2*(1+math.cos((j-aspect_end)/context_len)))
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise gaussian mask function:
# exp(-(aspect_start-j)^2/context_len^2)    if j<aspect_start
# 0                                         if aspect_start<=j<=aspect_end
# exp(-(j-aspect_end)^2/context_len^2)      if aspect_end<j<text_len
# 0                                         if text_len<=j<seq_len
def piecewise_gaussian_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(math.exp(-(aspect_start-j)**2/context_len**2))
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(math.exp(-(j-aspect_end)**2/context_len**2))
    for j in range(text_leni, seq_len):
        weights.append(0)

# position awareness using the piecewise sqrt mask function:
# 1/sqrt(aspect_start-j)    if j<aspect_start
# 0                         if aspect_start<=j<=aspect_end
# 1/sqrt(j-aspect_end)      if aspect_end<j<text_len
# 0                         if text_len<=j<seq_len
def piecewise_sqrt_mask(aspect_start, aspect_end, context_len, seq_len, max_seq_len, text_leni, weights):
    for j in range(aspect_start):
        weights.append(1/math.sqrt(aspect_start-j))
    for j in range(aspect_start, min(aspect_end+1,max_seq_len)):
        weights.append(0)
    for j in range(aspect_end+1, text_leni):
        weights.append(1/math.sqrt(j-aspect_end))
    for j in range(text_leni, seq_len):
        weights.append(0)