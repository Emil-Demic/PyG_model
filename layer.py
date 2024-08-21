import torch
# from torch.nn import Linear
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F


class GNNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.W = Linear(self.in_feats,  self.out_feats, bias=False, weight_initializer='glorot')
        self.a = Linear(self.out_feats * 2, 1, bias=False)

    def forward(self, x, adj):
        n = x.shape[1]
        x = self.W(x)
        x_weighted = adj @ x
        # x_i = torch.repeat_interleave(x, torch.ones(n, dtype=torch.int32) * n, dim=1)
        x_i = torch.repeat_interleave(x, torch.ones(n, dtype=torch.int32, device="cuda:0") * n, dim=1)
        x_j = x.repeat((1, n, 1))
        x_concat = torch.cat((x_i, x_j), dim=2)
        e = self.a(x_concat)
        e = F.leaky_relu(e, 0.2)
        e = e.view(-1, n, n)
        alpha = F.softmax(e, dim=2)

        # TODO ??? PRABILONO ???
        x = alpha @ x
        x += x_weighted
        x = F.leaky_relu(x, 0.2)
        return x



