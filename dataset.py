import csv
import os
import random

import torch
from PIL import Image
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from torchvision.models import ResNeXt50_32X4D_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, CenterCrop, Normalize, Compose, ToImage, ToDtype, RGB

from torch_geometric.data import Data


def get_boxes(path, width, height):
    boxes = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            box = torch.tensor([float(x) for x in row[2:]])
            box[0] = box[0] / width * 224.
            box[2] = box[2] / width * 224.
            box[1] = box[1] / height * 224.
            box[3] = box[3] / height * 224.
            boxes.append(box)

    if len(boxes) == 0:
        boxes.append(torch.zeros((1, 4)))

    if len(boxes) > 1:
        boxes = torch.stack(boxes)
    else:
        boxes = boxes[0].unsqueeze(0)

    return boxes


class TripletData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_a':
            return self.x_a.size(0)
        if key == 'edge_index_p':
            return self.x_p.size(0)
        if key == 'edge_index_n':
            return self.x_n.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class DatasetTrain(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetTrain, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        csv_files_sketch = os.listdir("train/sketch/GraphFeatures/")
        jpg_files_sketch = os.listdir("train/sketch/Image/")
        csv_files_image = os.listdir("train/image/GraphFeatures/")
        jpg_files_image = os.listdir("train/image/Image/")
        return csv_files_sketch + jpg_files_sketch + csv_files_image + jpg_files_image

    @property
    def processed_file_names(self):
        return 'processed_train.pt'

    def process(self):
        data_list = []

        file_list = os.listdir("train/sketch/GraphFeatures/")
        file_list = [x.split(".")[0] for x in file_list[:-30]]
        csv_files_sketch = "train/sketch/GraphFeatures/"
        jpg_files_sketch = "train/sketch/Image/"
        csv_files_image = "train/image/GraphFeatures/"
        jpg_files_image = "train/image/Image/"

        weights = ResNeXt50_32X4D_Weights.DEFAULT
        preprocess_image = weights.transforms()
        preprocess_sketch = Compose([
            RGB(),
            Resize(232, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(224),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for file in file_list:
            input_image = Image.open(os.path.join(jpg_files_sketch, file + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_sketch(input_image)
            x = get_boxes(os.path.join(csv_files_sketch, file + ".csv"), width, height)
            num_nodes = x.shape[0]
            adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            data_a = Data(x=x, edge_index=adj_matrix, img=input_image)

            input_image = Image.open(os.path.join(jpg_files_image, file + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_image(input_image)
            x = get_boxes(os.path.join(csv_files_image, file + ".csv"), width, height)
            num_nodes = x.shape[0]
            adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            data_p = Data(x=x, edge_index=adj_matrix, img=input_image)

            neg_sample = file
            while neg_sample == file:
                neg_sample = file_list[random.randint(0, len(file_list) - 1)]

            input_image = Image.open(os.path.join(jpg_files_image, neg_sample + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_image(input_image)
            x = get_boxes(os.path.join(csv_files_image, neg_sample + ".csv"), width, height)
            num_nodes = x.shape[0]
            adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            data_n = Data(x=x, edge_index=adj_matrix, img=input_image)

            data = TripletData(x_a=data_a.x, edge_index_a=data_a.edge_index, img_a=data_a.img,
                               x_p=data_p.x, edge_index_p=data_p.edge_index, img_p=data_p.img,
                               x_n=data_n.x, edge_index_n=data_n.edge_index, img_n=data_n.img,)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


class DatasetSketchTest(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetSketchTest, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        csv_files = os.listdir("test/sketch/GraphFeatures/")
        jpg_files = os.listdir("test/sketch/Image/")
        return csv_files + jpg_files

    @property
    def processed_file_names(self):
        return 'processed_sketch_test.pt'

    def process(self):
        data_list = []

        file_list = os.listdir("test/sketch/GraphFeatures/")
        file_list = [x.split(".")[0] for x in file_list[-30:]]
        csv_files_sketch = "test/sketch/GraphFeatures"
        jpg_files_sketch = "test/sketch/Image"

        preprocess_sketch = Compose([
            RGB(),
            Resize(232, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(224),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for file in file_list:
            input_image = Image.open(os.path.join(jpg_files_sketch, file + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_sketch(input_image)
            x = get_boxes(os.path.join(csv_files_sketch, file + ".csv"), width, height)
            num_nodes = x.shape[0]
            adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            data = Data(x=x, edge_index=adj_matrix, img=input_image)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


class DatasetImageTest(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetImageTest, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        csv_files = os.listdir("test/image/GraphFeatures/")
        jpg_files = os.listdir("test/image/Image/")
        return csv_files + jpg_files

    @property
    def processed_file_names(self):
        return 'processed_image_test.pt'

    def process(self):
        data_list = []

        file_list = os.listdir("test/image/GraphFeatures/")
        file_list = [x.split(".")[0] for x in file_list[-30:]]
        csv_files_image = "test/image/GraphFeatures/"
        jpg_files_image = "test/image/Image/"

        weights = ResNeXt50_32X4D_Weights.DEFAULT
        preprocess_image = weights.transforms()

        for file in file_list:
            input_image = Image.open(os.path.join(jpg_files_image, file + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_image(input_image)
            x = get_boxes(os.path.join(csv_files_image, file + ".csv"), width, height)
            num_nodes = x.shape[0]
            adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            data = Data(x=x, edge_index=adj_matrix, img=input_image)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
