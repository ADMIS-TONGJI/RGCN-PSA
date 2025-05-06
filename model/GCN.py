# Graph Convolution Network

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, args, num_nodes):
        super(GCN, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.weights = nn.Parameter(torch.FloatTensor(num_nodes, self.args.cheb_k, self.args.dim_short_hidden, self.args.dim_gcn_hidden))
        self.bias = nn.Parameter(torch.FloatTensor(num_nodes, self.args.dim_gcn_hidden))
        self.G1 = nn.Parameter(torch.FloatTensor(num_nodes, self.args.dim_graph))
        
    def forward(self, x):
        G = torch.softmax(F.relu(torch.matmul(self.G1, self.G1.T)), dim=-1)
        G_set = [torch.eye(self.num_nodes).reshape(self.num_nodes,self.num_nodes).to(self.G1.device), G]
        for k in range(2, self.args.cheb_k):
            G_set.append(torch.matmul(2 * G , G_set[-1]) - G_set[-2])
        G = torch.stack(G_set, dim=0)
        x_g = torch.einsum("knm,bmi->bkni", G, x)      #B, cheb_k, N, dim_gcn_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_gcn_in
        x_gconv = torch.einsum('bnki,nkih->bnh', x_g, self.weights) + self.bias     #b, N, dim_gcn_hidden
        return x_gconv, self.G1