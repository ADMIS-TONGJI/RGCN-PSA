import torch
import torch.nn as nn

class SineActivation(nn.Module):
    def __init__(self, dim_in, dim_out, node):
        super(SineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(dim_in, 1), requires_grad=True)
        self.b0 = nn.parameter.Parameter(torch.randn(node, 1), requires_grad=True)
        self.w = nn.parameter.Parameter(torch.randn(dim_in, dim_out - 1), requires_grad=True)
        self.b = nn.parameter.Parameter(torch.randn(node, dim_out - 1), requires_grad=True)
    
    def forward(self, x):
        # 初始化权重和偏置
        v0 = torch.matmul(x, self.w0) + self.b0
        v = torch.sin(torch.matmul(x, self.w) + self.b)
        x = torch.cat([v0, v], 2)
        return x


class TimeEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, node):
        super(TimeEncoder, self).__init__()
        self.ls = SineActivation(dim_in, dim_out, node)

    def forward(self, x):
        x = self.ls(x) #bs*n*dim_out
        return x