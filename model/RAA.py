# Regional and Aggregation Analysis

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TimeEncoder import TimeEncoder
from model.GCN import GCN
from model.SelfAttention import Self_Attention_Muti_Head

class RAA(nn.Module):
    def __init__(self, args, adj):
        super(RAA, self).__init__()
        self.args = args
        self.adj = adj

        self.soft_clusters = nn.Parameter(torch.FloatTensor(self.args.num_clusters, self.args.num_nodes))
        
        self.encoder = TimeEncoder(self.args.daily_window, self.args.dim_encoder_hidden, self.args.num_nodes)
        self.attention = Self_Attention_Muti_Head(self.args.dim_encoder_hidden,self.args.dim_k,self.args.dim_v,self.args.nums_head)
        self.fc_short_hidden = nn.Linear(self.args.dim_v, self.args.dim_short_hidden)

        self.gcn_clusters = GCN(self.args, self.args.num_clusters)
        self.fc_clusters = nn.Linear(self.args.dim_gcn_hidden, self.args.dim_clusters_hidden)
        
    def forward(self, x):
        encoder_out = self.encoder(x) #b,n,dim_encoder_hidden
        attention_out = self.attention(encoder_out)
        y_short_hidden = F.relu(self.fc_short_hidden(attention_out))
        
        gcn_clusters_out, clusters_G = self.gcn_clusters(torch.einsum("cn,bnd->bcd", self.soft_clusters, y_short_hidden)) #b,c,d
        y_clusters_hidden = torch.einsum("cn,bcd->bnd", self.soft_clusters, gcn_clusters_out)
        y_clusters = F.relu(self.fc_clusters(y_clusters_hidden))

        y_short_out = torch.cat((y_short_hidden, y_clusters), dim=-1)

        return y_short_out, self.soft_clusters, clusters_G