import torch
from torch.nn import Sequential, Identity
from torch_geometric.nn.dense import DenseGATConv, DenseSAGEConv
from torch_geometric.utils import to_dense_batch
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d
from torchvision.ops import roi_align


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DenseGATConv(512, 512)
        self.fc1 = torch.nn.Linear(4096, 512)
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
        global_features = torch.repeat_interleave(global_features, torch.bincount(batch), dim=0)
        x = torch.concat((x, global_features), dim=1)
        x = self.fc1(x)
        x = to_dense_batch(x, batch)
        x = self.conv1(x[0], edge_index)
        x = torch.mean(x[0], dim=1)
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
