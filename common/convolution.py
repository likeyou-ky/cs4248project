import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True, isfText=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_f_text = isfText
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        print(f'W CREATION location: {"Cuda" if self.weight.is_cuda else "CPU"}')
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        print(f'TEXT location: {"Cuda" if text.is_cuda else "CPU"}')
        print(f'W location: {"Cuda" if self.weight.is_cuda else "CPU"}')
        if self.is_f_text:
            text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output