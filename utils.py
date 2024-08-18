import scipy.spatial.distance as ssd
import os


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
    top20 = 0
    tmpLine = ""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[:5]:
            top5 = top5 + 1
        if i in rank[:10]:
            top10 = top10 + 1
        if i in rank[:20]:
            top20 = top20 + 1
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
    return top1, top5, top10, top20


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
