#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:19:03 2019

@author: tpetit
"""
import torch
from torch.utils import data
import numpy as np
import cv2, dlib
import os


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

class Dataset(data.Dataset ) :
    
    def __init__(self, list_IDs, labels, datapath):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.datapath = datapath
        #self.conf_threshold = 0.97
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = cv2.imread(os.path.join(self.datapath, ID))
        
        X = torch.from_numpy(X)
        X = (X.permute(2, 1, 0).float()-127.5)/128
        y = self.labels[ID]
        return X, y


class BalancedBatchSampler(data.BatchSampler):
    """
    from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, partition, labels, n_classes = 10, n_samples = 25):
        self.labels = [labels[x] for x in partition]
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where([l == label for l in self.labels])[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes
        self.mean = None

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.labels):
            if self.mean == None :
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            else :
                classes = list(np.random.choice(self.labels_set, 1))
                seed = classes[0]
                closest = sorted(self.mean.keys(), key=lambda k: np.linalg.norm(self.mean[k]-self.mean[seed]))
                closest.remove(seed)
                classes += [c for c in closest[:self.n_classes-1]]
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.labels) // self.batch_size

    def set_mean(self, mean):
        self.mean = mean 


#class BalancedClassesSampler(data.BatchSampler): # custom
#    """
#    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
#    Returns batches of size n_classes * n_samples
#    """
#
#    def __init__(self, partition, labels, n_classes = 3, n_samples = 10):
#        self.labels = [labels[x] for x in partition]
#        self.labels_set = list(set(self.labels))
#        self.label_to_indices = {label: np.where([l == label for l in self.labels])[0]
#                                 for label in self.labels_set}
#        for l in self.labels_set:
#            np.random.shuffle(self.label_to_indices[l])
#        self.used_label_indices_count = {label: 0 for label in self.labels_set}
#        self.count = 0
#        self.n_classes = n_classes
#        self.n_samples = n_samples
#        self.batch_size = self.n_samples * self.n_classes
#
#    def __iter__(self):
#        self.count = 0
#        while self.count + self.batch_size < len(self.labels):
#            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
#            indices = []
#            for class_ in classes:
#                class_sample = []
#                class_sample.extend(self.label_to_indices[class_][
#                               self.used_label_indices_count[class_]:self.used_label_indices_count[
#                                                                         class_] + self.n_samples])
#                self.used_label_indices_count[class_] += self.n_samples
#                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
#                    np.random.shuffle(self.label_to_indices[class_])
#                    self.used_label_indices_count[class_] = 0
#                indices.append(class_sample)
#            yield indices
#            self.count += self.n_classes * self.n_samples
#
#    def __len__(self):
#        return len(self.labels) // self.batch_size
