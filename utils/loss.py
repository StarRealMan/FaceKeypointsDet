import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def genGTHeatmaps(keyPoint):

    if keyPoint != None:
        
        gtHeatmap = torch.zeros((96, 96))
        coorPos = torch.tensor([keyPoint[0], keyPoint[1]])

        for i in range(gtHeatmap.shape[0]):
            for j in range(gtHeatmap.shape[1]):
                thisPos = torch.tensor([j, i], dtype = torch.float)
                dist = torch.dist(thisPos, coorPos)
                if dist < 1:
                    gtHeatmap[i][j] = 1
                elif dist < 2:
                    gtHeatmap[i][j] = 0.8
                else:
                    gtHeatmap[i][j] = 1/dist

    else:
        gtHeatmap = None

    return gtHeatmap

def calLoss(batchHeatmap, batchAnnolist, alpha = 2, beta = 4):

    lossSum = 0

    for batch in range(batchHeatmap.shape[0]):

        annoList = batchAnnolist[batch]
        loss = 0
        valid = 0

        for kpID, keypoint in enumerate(batchAnnolist[batch]):
            
            if keypoint != None: 
                heatMap = batchHeatmap[batch][kpID]
                GTHeatmap = genGTHeatmaps(keypoint)

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

        loss = loss / valid
        lossSum = lossSum + loss

    return lossSum

if __name__ == "__main__":

    annoList = [[(65.0570526316, 34.9096421053), (30.9037894737, 34.9096421053), None,
                 None, (37.6781052632, 36.3209684211), (24.9764210526, 36.6032210526),
                 (55.7425263158, 27.5709473684), None, (42.1938947368, 28.1354526316),
                 (16.7911578947, 32.0871157895), None, (60.8229473684, 73.0143157895),
                 (33.7263157895, 72.732), (47.2749473684, 70.1917894737), None]]

    heatmap = torch.randn((1, 1, 96, 96))
    heatmap = torch.sigmoid(heatmap)
    
    loss = calLoss(heatmap, annoList)

    print(loss)

    # GTheatmap = genGTHeatmaps(annoList[0][0])

    # GTheatmap = GTheatmap.detach().numpy()
    # print(GTheatmap.shape)

    # plt.imshow(GTheatmap)
    # plt.show()