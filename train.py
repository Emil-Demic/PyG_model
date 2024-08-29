import numpy as np
import torch
import tqdm
from torch.nn import TripletMarginLoss
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from dataset import DatasetTrain, DatasetSketchTest, DatasetImageTest
from model import TripletModel
from utils import compute_view_specific_distance, calculate_accuracy, calculate_accuracy_alt
from config import args

dataset = DatasetTrain("data")
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, follow_batch=['x_a', 'x_p', 'x_n'])

dataset_sketch_test = DatasetSketchTest("data")
dataset_image_test = DatasetImageTest("data")
loader_sketch_test = DataLoader(dataset_sketch_test, batch_size=args.batch_size * 3, shuffle=False)
loader_image_test = DataLoader(dataset_image_test, batch_size=args.batch_size * 3, shuffle=False)

model = TripletModel()
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# scheduler = lr_scheduler.StepLR(optimizer, args.lr_scheduler_step, gamma=0.1, last_epoch=-1)
loss = TripletMarginLoss(margin=0.2)
if args.cuda:
    loss.cuda()


best_score = (0.0, 0.0)
for epoch in range(args.epochs):
    running_loss = 0.0
    model.train()
    for i, batch in enumerate(loader):
        if args.cuda:
            batch.cuda()
        batch.img_a = batch.img_a.view(-1, 3, 224, 224)
        batch.img_p = batch.img_p.view(-1, 3, 224, 224)
        batch.img_n = batch.img_n.view(-1, 3, 224, 224)

        optimizer.zero_grad()

        out = model(batch)

        epoch_loss = loss(out[0], out[1], out[2])
        running_loss += epoch_loss.item()

        epoch_loss.backward()
        optimizer.step()

        if i % 5 == 4:
            print(f'[{epoch:03d}, {i:03d}] loss: {running_loss / 5:0.5f}')
            running_loss = 0.0
    # scheduler.step()

    with torch.no_grad():
        model.eval()
        sketch_out_list = []
        for batch in tqdm.tqdm(loader_sketch_test):
            batch.cuda()
            batch.img = batch.img.view(-1, 3, 224, 224)
            out = model.get_embedding(batch, True)
            sketch_out_list.append(out.cpu().numpy())

        image_out_list = []
        for batch in tqdm.tqdm(loader_image_test):
            batch.cuda()
            batch.img = batch.img.view(-1, 3, 224, 224)
            out = model.get_embedding(batch, False)
            image_out_list.append(out.cpu().numpy())

        sketch_out_list = np.concatenate(sketch_out_list)
        image_out_list = np.concatenate(image_out_list)

        dis = compute_view_specific_distance(sketch_out_list, image_out_list)

        num = dis.shape[0]
        top1, top5, top10, top20 = calculate_accuracy(dis)
        print(str(epoch + 1) + ':  top1: ' + str(top1 / float(num)))
        print(str(epoch + 1) + ':  top5: ' + str(top5 / float(num)))
        print(str(epoch + 1) + ': top10: ' + str(top10 / float(num)))
        print(str(epoch + 1) + ': top20: ' + str(top20 / float(num)))
        print("top1, top5, top10:", top1, top5, top10)

        top1, top5, top10, meanK = calculate_accuracy_alt(sketch_out_list, image_out_list)
        print("top1, top5, top10, meanK:", top1, top5, top10, meanK)

        if top1 > best_score[0] and top10 > best_score[1]:
            print("saving model")
            best_score = (top1, top10)
            torch.save(model.state_dict(), "model/model_" + str(epoch) + ".pth")
            with open("model/result_" + str(epoch) + ".txt", "w") as f:
                f.write(f"top1: {top1}, top5: {top5}, top10: {top10}, meanK: {meanK}")

