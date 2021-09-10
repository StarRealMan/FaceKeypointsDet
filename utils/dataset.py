import csv
import sys

from numpy.lib.type_check import imag
sys.path.append("..")
import os
import numpy as np
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

class FaceDataset(data.Dataset):

    def __init__(self, datapath = "../data/", split = "train"):
        
        if split == "train":
            split_path = "training.csv"
            self.anno = []
        else:
            split_path = "test.csv"

        path = os.path.join(datapath, split_path)

        self.data = []

        with open(path,'rt') as f: 
            csvReader = csv.reader(f)
            for num, row in tqdm(enumerate(csvReader)):
                if split == "train":
                    if num > 0:
                        keypoints = []
                        for pointNum in range(15):
                            if row[pointNum * 2] != '':
                                keypoints.append((float(row[pointNum * 2]), float(row[pointNum * 2 + 1])))
                            else:
                                keypoints.append((-1, -1))
                        self.anno.append(copy.deepcopy(np.array(keypoints)))
                        keypoints.clear()
                        image = np.array(list(map(int, row[30].split(' '))), dtype = np.uint8).reshape((96, 96))
                        image = np.expand_dims(image, axis = 0)
                        self.data.append(image)
                elif split == "test":
                    if num > 0:
                        image = np.array(list(map(int, row[30].split(' '))), dtype = np.uint8).reshape((96, 96))
                        image = np.expand_dims(image, axis = 0)
                        self.data.append(image)
                # if num > 10:
                #     break

        self.split = split

    def __getitem__(self, index):
        if self.split == "train":
            data = self.data[index]
            anno = self.anno[index]

        elif self.split == "test":
            data = self.data[index]
            anno = None

        return data, anno

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    dataset = FaceDataset()
    dataloader = data.DataLoader(dataset, shuffle = True, batch_size = 3, \
                                num_workers = 8, drop_last = False)
    
    for i, data in enumerate(dataloader):

        image, anno = data

        image = image[0][0].detach().numpy()
        # cv2.imshow("face", image)
        plt.imshow(image)

        print(anno.shape)

        break

    # cv2.waitKey(0)
    plt.show()