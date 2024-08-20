import torch
from torch.nn import Sequential, Identity, Linear, Parameter
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGATConv
from torch_geometric.nn.conv import TransformerConv, GATConv
from torch_geometric.utils import to_dense_batch
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights
from torchvision.ops import roi_align

from layer import GNNLayer


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        self.feature_extractor = Sequential(*list(model.children())[:-2])
        self.feature_extractor.classifier = Identity()
        self.pool_method = torch.nn.AdaptiveMaxPool2d(1)

    def forward(self, x, edge_index, img, batch, sketch=True):
        x = self.feature_extractor(img)
        x = self.pool_method(x).view(-1, 3024)
        return F.normalize(x)


class TripletModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Model1()

    def forward(self, data):
        res1 = self.embedding(data.x_a, data.edge_index_a, data.img_a, data.x_a_batch, True)
        res2 = self.embedding(data.x_p, data.edge_index_p, data.img_p, data.x_p_batch, False)
        res3 = self.embedding(data.x_n, data.edge_index_n, data.img_n, data.x_n_batch, False)
        return res1, res2, res3

    def get_embedding(self, data, sketch=True):
        return self.embedding(data.x, data.edge_index, data.img, data.batch, sketch)
