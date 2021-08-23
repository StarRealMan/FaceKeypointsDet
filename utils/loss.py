import torch
import torch.nn as nn
import torch.nn.functional as F

def genGTHeatmaps(annoList):

    GTHeatmaps = []

    for keyPoint in annoList:
        gtHeatmap = torch.zeros((96, 96), dtype = torch.long)

        coorX = keyPoint[0]
        coorY = keyPoint[1]

        

        GTHeatmaps.append(gtHeatmap)

    return torch.tensor(GTHeatmaps)


def calLoss():
    pass



if __name__ == "__main__":
