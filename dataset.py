import csv
import math
import os
import random

import torch
from PIL import Image
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.utils import dense_to_sparse
from torchvision.models import ResNeXt50_32X4D_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, CenterCrop, Normalize, Compose, ToImage, ToDtype, RGB, ToTensor

from torch_geometric.data import Data


def get_boxes(path, width, height, is_sketch=True):
    boxes = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            box = torch.tensor([float(x) for x in row[2:]])
            if is_sketch:
                box = box * (224. / 256.)
            else:
                box[0] = box[0] * (224. / width)
                box[2] = box[2] * (224. / width)
                box[1] = box[1] * (224. / height)
                box[3] = box[3] * (224. / height)
            boxes.append(box)

    if len(boxes) == 0:
        boxes.append(torch.zeros(4))

    # boxes.insert(0, torch.zeros(4))

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


class DatasetTrain(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetTrain, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        csv_files_sketch = os.listdir("train/sketch/GraphFeatures/")
        jpg_files_sketch = os.listdir("train/sketch/Image/")
        csv_files_image = os.listdir("train/image/GraphFeatures/")
        jpg_files_image = os.listdir("train/image/Image/")
        return csv_files_sketch + jpg_files_sketch + csv_files_image + jpg_files_image

    @property
    def processed_file_names(self):
        processed_files_sketches = []
        processed_files_images = []
        for i in range(len(self.raw_file_names) // 4):
            processed_files_sketches.append(f"data_sketch_train_{i}.pt")
            processed_files_images.append(f"data_image_train_{i}.pt")
        return processed_files_sketches + processed_files_images

    def process(self):
        idx = 0

        file_list = os.listdir("train/sketch/GraphFeatures/")
        file_list = [x.split(".")[0] for x in file_list]
        csv_files_sketch = "train/sketch/GraphFeatures/"
        jpg_files_sketch = "train/sketch/Image/"
        csv_files_image = "train/image/GraphFeatures/"
        jpg_files_image = "train/image/Image/"

        preprocess_image = Compose([
            Resize((224, 224)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        preprocess_sketch = preprocess_image
        # preprocess_sketch = Compose([
        #     RGB(),
        #     Resize(224, interpolation=InterpolationMode.BILINEAR),
        #     ToImage(),
        #     ToDtype(torch.float32, scale=True),
        #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        for file in file_list:
            input_image = Image.open(os.path.join(jpg_files_sketch, file + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_sketch(input_image)
            x = get_boxes(os.path.join(csv_files_sketch, file + ".csv"), width, height, True)
            num_nodes = x.shape[0]
            # adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            centres = x.view(-1, 2, 2).mean(dim=1)
            adj_matrix = torch.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    dist = torch.linalg.norm(centres[i, :] - centres[j, :])
                    adj_matrix[i, j] = dist
                    adj_matrix[j, i] = dist
            adj_matrix = adj_matrix / math.sqrt(height ** 2 + width ** 2)
            adj_matrix = 1 - adj_matrix
            adj_matrix[adj_matrix < 0] = 0
            adj_matrix = dense_to_sparse(adj_matrix)
            data_s = Data(x=x, edge_index=adj_matrix[0], edge_attr=adj_matrix[1], img=input_image)

            input_image = Image.open(os.path.join(jpg_files_image, file + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_image(input_image)
            x = get_boxes(os.path.join(csv_files_image, file + ".csv"), width, height, False)
            num_nodes = x.shape[0]
            # adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            centres = x.view(-1, 2, 2).mean(dim=1)
            adj_matrix = torch.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    dist = torch.linalg.norm(centres[i, :] - centres[j, :])
                    adj_matrix[i, j] = dist
                    adj_matrix[j, i] = dist
            adj_matrix = adj_matrix / torch.sqrt(torch.tensor(256 ** 2 + 256 ** 2))
            adj_matrix = 1 - adj_matrix
            adj_matrix[adj_matrix < 0] = 0
            adj_matrix = dense_to_sparse(adj_matrix)
            data_i = Data(x=x, edge_index=adj_matrix[0], edge_attr=adj_matrix[1], img=input_image)

            torch.save(data_s, os.path.join(self.processed_dir, f'data_sketch_train_{idx}.pt'))
            torch.save(data_i, os.path.join(self.processed_dir, f'data_image_train_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.raw_file_names) // 4

    def get(self, idx):
        data_a = torch.load(os.path.join(self.processed_dir, f'data_sketch_train_{idx}.pt'))
        data_p = torch.load(os.path.join(self.processed_dir, f'data_image_train_{idx}.pt'))

        negative_idx = random.randint(0, self.len() - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, self.len() - 1)

        data_n = torch.load(os.path.join(self.processed_dir, f'data_image_train_{negative_idx}.pt'))

        data = TripletData(x_a=data_a.x, edge_index_a=data_a.edge_index, edge_attr_a=data_a.edge_attr, img_a=data_a.img,
                           x_p=data_p.x, edge_index_p=data_p.edge_index, edge_attr_p=data_p.edge_attr, img_p=data_p.img,
                           x_n=data_n.x, edge_index_n=data_n.edge_index, edge_attr_n=data_n.edge_attr, img_n=data_n.img)
        return data


class DatasetSketchTest(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetSketchTest, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        csv_files = os.listdir("test/sketch/GraphFeatures/")
        jpg_files = os.listdir("test/sketch/Image/")
        return csv_files + jpg_files

    @property
    def processed_file_names(self):
        processed_files_sketches = []
        for i in range(len(self.raw_file_names) // 2):
            processed_files_sketches.append(f"data_sketch_test_{i}.pt")
        return processed_files_sketches

    def process(self):
        idx = 0

        file_list = os.listdir("train/sketch/GraphFeatures/")
        file_list = [x.split(".")[0] for x in file_list]
        csv_files_sketch = "train/sketch/GraphFeatures/"
        jpg_files_sketch = "train/sketch/Image/"

        preprocess_sketch = Compose([
            Resize((224, 224)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for file in file_list:
            input_image = Image.open(os.path.join(jpg_files_sketch, file + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_sketch(input_image)
            x = get_boxes(os.path.join(csv_files_sketch, file + ".csv"), width, height, True)
            num_nodes = x.shape[0]
            # adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            centres = x.view(-1, 2, 2).mean(dim=1)
            adj_matrix = torch.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    dist = torch.linalg.norm(centres[i, :] - centres[j, :])
                    adj_matrix[i, j] = dist
                    adj_matrix[j, i] = dist
            adj_matrix = adj_matrix / torch.sqrt(torch.tensor(256 ** 2 + 256 ** 2))
            adj_matrix = 1 - adj_matrix
            adj_matrix[adj_matrix < 0] = 0
            adj_matrix = dense_to_sparse(adj_matrix)
            data_s = Data(x=x, edge_index=adj_matrix[0], edge_attr=adj_matrix[1], img=input_image)

            torch.save(data_s, os.path.join(self.processed_dir, f'data_sketch_test_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_sketch_test_{idx}.pt'))
        return data


class DatasetImageTest(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetImageTest, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        csv_files = os.listdir("test/image/GraphFeatures/")
        jpg_files = os.listdir("test/image/Image/")
        return csv_files + jpg_files

    @property
    def processed_file_names(self):
        processed_files_images = []
        for i in range(len(self.raw_file_names) // 2):
            processed_files_images.append(f"data_image_test_{i}.pt")
        return processed_files_images

    def process(self):
        idx = 0

        file_list = os.listdir("train/sketch/GraphFeatures/")
        file_list = [x.split(".")[0] for x in file_list]
        csv_files_image = "train/image/GraphFeatures/"
        jpg_files_image = "train/image/Image/"

        preprocess_image = Compose([
            Resize((224, 224)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for file in file_list:
            input_image = Image.open(os.path.join(jpg_files_image, file + ".jpg"))
            width, height = input_image.size
            input_image = preprocess_image(input_image)
            x = get_boxes(os.path.join(csv_files_image, file + ".csv"), width, height, False)
            num_nodes = x.shape[0]
            # adj_matrix = dense_to_sparse(torch.ones((num_nodes, num_nodes)))[0]
            centres = x.view(-1, 2, 2).mean(dim=1)
            adj_matrix = torch.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    dist = torch.linalg.norm(centres[i, :] - centres[j, :])
                    adj_matrix[i, j] = dist
                    adj_matrix[j, i] = dist
            adj_matrix = adj_matrix / torch.sqrt(torch.tensor(256 ** 2 + 256 ** 2))
            adj_matrix = 1 - adj_matrix
            adj_matrix[adj_matrix < 0] = 0
            adj_matrix = dense_to_sparse(adj_matrix)
            data_i = Data(x=x, edge_index=adj_matrix[0], edge_attr=adj_matrix[1], img=input_image)

            torch.save(data_i, os.path.join(self.processed_dir, f'data_image_test_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_image_test_{idx}.pt'))
        return data
