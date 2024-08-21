import torch
from torch.nn import Sequential, Identity, Linear, Parameter
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGATConv
from torch_geometric.nn.conv import TransformerConv, GATConv
from torch_geometric.utils import to_dense_batch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, regnet_y_8gf, RegNet_Y_8GF_Weights
from torchvision.ops import roi_align

from layer import GNNLayer


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = regnet_y_8gf(weights=RegNet_Y_8GF_Weights.DEFAULT)
        self.feature_extractor = Sequential(*list(model.children())[:-2])
        self.pool_method = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, img):
        x = self.feature_extractor(img)
        x = self.pool_method(x).view(-1, 2016)
        return F.normalize(x)


class TripletModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Model1()

    def forward(self, data):
        res1 = self.embedding(data.img_a)
        res2 = self.embedding(data.img_p)
        res3 = self.embedding(data.img_n)
        return res1, res2, res3

    def get_embedding(self, data, sketch=True):
        return self.embedding(data.img)
