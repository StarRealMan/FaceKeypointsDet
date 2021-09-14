import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, input):
        
        output = self.main(input)
        output = output + input
        output = F.relu(output, True)

        return output

class DownSampleResBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 2, 1, 1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, 1),
            nn.BatchNorm2d(channels * 2)
        )

        self.side = nn.Conv2d(channels, channels * 2, 1, 2, 0, 1)

    def forward(self, input):
        
        output = self.main(input)
        input = self.side(input)
        output = output + input
        output = F.relu(output, True)

        return output


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(channels, int(channels/2), 3, 2, 1, 1),
            nn.BatchNorm2d(int(channels/2)),
            nn.ReLU(True),
            nn.Conv2d(int(channels/2), int(channels/4), 3, 1, 1, 1),
            nn.BatchNorm2d(int(channels/4)),
            nn.ReLU(True)
        )

    def forward(self, input):
        
        output = self.main(input)

        return output


class FaceKeypointModel(nn.Module):

    def __init__(self, in_chan = 1, chan_mag = 64, layers = 4):
        super(FaceKeypointModel, self).__init__()
        self.fieldconv = nn.Conv2d(in_chan, chan_mag, 7, 1, 3, 1)

        self.layers = layers

        self.resblockList = []

        self.resblockList.append(ResBlock(chan_mag).cuda())
        self.resblockList.append(ResBlock(chan_mag).cuda())

        for layerNum in range(layers - 1):
            self.resblockList.append(DownSampleResBlock(chan_mag * 2 ** layerNum).cuda())
            self.resblockList.append(ResBlock(chan_mag * 2 ** (layerNum + 1)).cuda())

        self.encoder = nn.Sequential(
            nn.Conv2d(chan_mag * 2 ** (layers - 1), chan_mag * 2 ** (layers - 1), 3, 1, 1, 1),
            nn.ReLU(True)
        )

        self.upsampleblockList = []
        self.upsampleblockList.append(UpSampleBlock(chan_mag * 2 ** (layers - 1)).cuda())
        lastChannel = chan_mag * 2 ** (layers - 1)

        for layerNum in range(layers - 2, 0, -1):
            self.upsampleblockList.append(UpSampleBlock(int(lastChannel/4) + chan_mag * 2 ** layerNum).cuda())
            lastChannel = int(lastChannel/4) + chan_mag * 2 ** layerNum

        lastChannel = int(lastChannel/4) + chan_mag

        self.decoder = nn.Sequential(
            nn.Conv2d(lastChannel, chan_mag, 3, 1, 1, 1),
            nn.BatchNorm2d(int(chan_mag)),
            nn.ReLU(True),
            nn.Conv2d(chan_mag, 15, 3, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        
        output = self.fieldconv(input)

        featSkipList = []

        for layer in range(self.layers):
            output = self.resblockList[layer * 2](output)
            output = self.resblockList[layer * 2 + 1](output)
            featSkipList.append(output)

        output = self.encoder(output)

        for layer, upsampleblock in enumerate(self.upsampleblockList):
            output = upsampleblock(output)
            output = torch.cat((output, featSkipList[self.layers - 2 - layer]), 1)

        output = self.decoder(output)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

if __name__ == "__main__":

    input = torch.randn((2, 1, 96, 96)).cuda()
    net = FaceKeypointModel()

    net.cuda()

    output = net(input)
    print(output.shape)

    visChan = 14
    outputNp = output[0][visChan].cpu().detach().numpy()
    plt.imshow(outputNp)
    plt.show()


