import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data

import utils.dataset as myDataset

def calLoss(batchHeatmap, batchAnnolist, batchGTHeatMapList, alpha = 2, beta = 4):

    lossSum = 0

    for batch in range(batchHeatmap.shape[0]):

        GTHeatMapList = batchGTHeatMapList[batch]
        annoList = batchAnnolist[batch]
        heatMapList = batchHeatmap[batch]
        loss = 0
        valid = 0

        for kpID, keypoint in enumerate(annoList):

            if keypoint[0] >= 0 and keypoint[1] >= 0:
                
                heatMap = heatMapList[kpID]
                GTHeatmap = GTHeatMapList[kpID]

                background = torch.zeros_like(heatMap)
                foreground = torch.where(GTHeatmap > 0.8, GTHeatmap, background)

                log_prob = (heatMap + 1e-7).log()
                one_prob = (torch.ones_like(heatMap) - heatMap)
                log_one_prob = (one_prob + 1e-7).log()
                one_GTHeatmap = (torch.ones_like(GTHeatmap) - GTHeatmap)
                
                loss_positive = foreground * one_prob ** alpha * log_prob
                loss_negative = one_GTHeatmap ** beta * heatMap ** alpha * log_one_prob

                loss = loss - loss_positive.sum() - loss_negative.sum()
                valid = valid + 1

            # print(- loss_positive.sum() - loss_negative.sum())

        loss = loss / valid
        lossSum = lossSum + loss

    return lossSum

if __name__ == "__main__":

    dataset = myDataset.FaceDataset(split = "train")
    dataloader = data.DataLoader(dataset, shuffle = True, batch_size = 32, \
                                 num_workers = 8, drop_last = False)

    for i, data in enumerate(dataloader):
        image, anno, gtmap = data

        heatmap = torch.randn((1, 15, 96, 96)).sigmoid()
        loss = calLoss(heatmap, anno, gtmap)
        print(loss)

        heatmap = gtmap
        loss = calLoss(heatmap, anno, gtmap)
        print(loss)

        break

    # annoList = [[(65.0570526316, 34.9096421053), (30.9037894737, 34.9096421053), (-1, -1),
    #              (-1, -1), (37.6781052632, 36.3209684211), (24.9764210526, 36.6032210526),
    #              (55.7425263158, 27.5709473684), (-1, -1), (42.1938947368, 28.1354526316),
    #              (16.7911578947, 32.0871157895), (-1, -1), (60.8229473684, 73.0143157895),
    #              (33.7263157895, 72.732), (47.2749473684, 70.1917894737), (-1, -1)]]

    # heatmap = torch.randn((1, 15, 96, 96)).cuda()
    # heatmap = torch.sigmoid(heatmap)
    
    # loss = calLoss(heatmap, annoList)

    # print(loss)


    # GTheatmap = genGTHeatmaps(annoList[0][0])

    # GTheatmap = GTheatmap.detach().numpy()
    # print(GTheatmap.shape)

    # plt.imshow(GTheatmap)
    # plt.show()