import scipy.spatial.distance as ssd
import os

import torch
import torch.nn.functional as F


def compute_view_specific_distance(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')

def outputHtml(sketchindex, indexList):
    imageNameList = os.listdir("test/sketch/Image")
    imageNameList = [x.split(".")[0] for x in imageNameList]
    sketchPath = "test/sketch/Image"
    imgPath = "test/image/Image"

    tmpLine = "<tr>"

    tmpLine += "<td><image src='%s' width=256 /></td>" % (
        os.path.join(sketchPath, str(imageNameList[sketchindex]).zfill(12) + ".png"))
    for i in indexList:
        if i != sketchindex:
            tmpLine += "<td><image src='%s' width=256 /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))
        else:
            tmpLine += "<td ><image src='%s' width=256   style='border:solid 2px red' /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))

    return tmpLine + "</tr>"

def calculate_accuracy(dist):
    top1 = 0
    top5 = 0
    top10 = 0
    tmpLine = ""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[:5]:
            top5 = top5 + 1
        if i in rank[:10]:
            top10 = top10 + 1
        tmpLine += outputHtml(i, rank[:10]) + "\n"

    htmlContent = """
       <html>
       <head></head>
       <body>
       <table>%s</table>
       </body>
       </html>""" % (tmpLine)
    with open(r"html_result/result.html", 'w+') as f:
        f.write(htmlContent)
    return top1, top5, top10

def calculate_accuracy_alt(query_feature_all, image_feature_all):
    query_feature_all = torch.tensor(query_feature_all)
    image_feature_all = torch.tensor(image_feature_all)

    rank = torch.zeros(len(query_feature_all))
    for idx, query_feature in enumerate(query_feature_all):
        distance = F.pairwise_distance(query_feature.unsqueeze(0), image_feature_all)
        target_distance = F.pairwise_distance(
            query_feature.unsqueeze(0), image_feature_all[idx].unsqueeze(0))
        rank[idx] = distance.le(target_distance).sum()

    rank1 = rank.le(1).sum().numpy() / rank.shape[0]
    rank5 = rank.le(5).sum().numpy() / rank.shape[0]
    rank10 = rank.le(10).sum().numpy() / rank.shape[0]
    rankM = rank.mean().numpy()

    return rank1, rank5, rank10, rankM


# def calculate_accuracy(dist):
#     top1 = 0
#     top5 = 0
#     top10 = 0
#     top20 = 0
#     for i in range(dist.shape[0]):
#         rank = dist[i].argsort()
#         if rank[0] == i:
#             top1 = top1 + 1
#         if i in rank[:5]:
#             top5 = top5 + 1
#         if i in rank[:10]:
#             top10 = top10 + 1
#         if i in rank[:20]:
#             top20 = top20 + 1
#
#     return top1, top5, top10, top20
