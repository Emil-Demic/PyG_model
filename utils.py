import scipy.spatial.distance as ssd


def compute_view_specific_distance(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')


def calculate_accuracy(dist):
    top1 = 0
    top5 = 0
    top10 = 0
    top20 = 0
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

    return top1, top5, top10, top20
