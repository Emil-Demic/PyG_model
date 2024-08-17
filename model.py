import torch
from torch.nn import Sequential, Identity, Linear, Parameter
import torch.nn.functional as F
# from torch_geometric.nn.dense import DenseGATConv, DenseSAGEConv, DenseGraphConv
from torch_geometric.utils import to_dense_batch
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d
from torchvision.ops import roi_align

from layer import GNNLayer


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_W1 = Linear(in_features=512, out_features=512)
        self.pool_W2 = Linear(in_features=512, out_features=512)
        self.conv1 = GNNLayer(4096, 512)
        model_s = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        self.feature_extractor_sketch = Sequential(*(list(model_s.children())[:-2]))
        model_i = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        self.feature_extractor_image = Sequential(*(list(model_i.children())[:-2]))
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, edge_index, img, batch, sketch=True):
        if sketch:
            extracted_features = self.feature_extractor_sketch(img)
        else:
            extracted_features = self.feature_extractor_image(img)
        global_features = self.global_pool(extracted_features)
        x = roi_align(extracted_features, x, spatial_scale=7. / 224., output_size=1)
        x = x.squeeze((2, 3))
        global_features = global_features.squeeze((2, 3))
        bincount = torch.bincount(batch)
        global_features = torch.repeat_interleave(global_features, bincount, dim=0)
        x = torch.concat((x, global_features), dim=1)
        x = to_dense_batch(x, batch)
        # non_zero = x[1].to(torch.int32).unsqueeze(2)
        # count = torch.count_nonzero(x[1], dim=1).to(torch.float)
        x = self.conv1(x[0], edge_index)
        # x = x * non_zero
        # x = F.sigmoid(self.pool_W1(x)) * (self.pool_W2(x))
        # x = torch.sum(x, dim=1, keepdim=False)
        # x = x / count.unsqueeze(1)
        x = torch.mean(x, dim=1)
        return x


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
