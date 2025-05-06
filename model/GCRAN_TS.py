# AttentionRNNTime

import torch
import torch.nn as nn
import torch.nn.functional as F
# import math

# from model.TimeEncoder import TimeEncoder
# from model.GCN import GCN
from model.RAA import RAA
from model.TSAttention import TSA

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.output = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        hidden_layer = self.relu(self.hidden(x))
        output_layer = self.output(hidden_layer)
        return output_layer


class GCRAN_TS(nn.Module):
    def __init__(self, args, adj_dataloader):
        super(GCRAN_TS, self).__init__()
        self.args = args
        self.adj_dataloader = adj_dataloader

        self.raa = RAA(self.args, self.adj_dataloader)
        self.tsa = TSA(self.args)
        self.end_conv = nn.Conv2d(1, self.args.horizon * self.args.dim_out, kernel_size=(1, args.dim_short_hidden+self.args.dim_clusters_hidden+self.args.dim_long_hidden), bias=True)
        self.pred = MLP(self.args.horizon+self.args.daily_window, self.args.horizon, self.args.horizon)

    def forward(self, x):
        x_short = x[:,:,-self.args.daily_window:] #bs*n*daily_window
        x_long = x[:,:,0:self.args.yearly_window*(self.args.shift_window*2+1)].reshape(x.shape[0],x.shape[1],self.args.yearly_window,-1).permute(0,1,3,2) #bs*n*(2*shift_window+1)*y
        y_short, soft_clusters, clusters_G = self.raa(x_short)
        y_long = self.tsa(x_long, y_short)
        y_hidden = torch.cat((y_short, y_long), dim=-1)
        y_conved = self.end_conv(y_hidden.unsqueeze(dim=1)).squeeze(dim=-1).permute(0,2,1) # b n t
        y_total = torch.cat((x_short, y_conved), dim=-1)
        y = self.pred(y_total)
        return y, soft_clusters, clusters_G
