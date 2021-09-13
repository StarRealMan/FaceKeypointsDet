import csv
import sys
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
            self.gtmaps = []
        else:
            split_path = "test.csv"

        path = os.path.join(datapath, split_path)

        self.data = []

        with open(path,'rt') as f: 
            csvReader = csv.reader(f)
            g = open("../data/gtmap.txt", "r")
            for num, row in tqdm(enumerate(csvReader)):
                if split == "train":
                    if num > 0:
                        keypoints = []
                        gtmap = []
                        for pointNum in range(15):
                            if row[pointNum * 2] != '':
                                keypoints.append((float(row[pointNum * 2]), float(row[pointNum * 2 + 1])))
                                line = g.readline()
                                gtmap.append(np.array(list(map(float, line[:-2].split(' '))), dtype = np.float32).reshape((96, 96)))
                            else:
                                keypoints.append((-1, -1))
                                line = g.readline()
                                gtmap.append(np.zeros((96, 96), dtype = float))
                        
                        self.anno.append(copy.deepcopy(np.array(keypoints)))
                        self.gtmaps.append(copy.deepcopy(np.array(gtmap)))
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
            gtmap = self.gtmaps[index]

        elif self.split == "test":
            data = self.data[index]
            anno = None
            gtmap = None

        return data, anno, gtmap

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    dataset = FaceDataset()
    dataloader = data.DataLoader(dataset, shuffle = True, batch_size = 3, \
                                num_workers = 8, drop_last = False)
    
    for i, data in enumerate(dataloader):

        image, anno, gtmap = data

        image = image[0][0].detach().numpy()
        plt.imshow(image)

        print(anno.shape)
        print(gtmap.shape)

        break

    plt.show()