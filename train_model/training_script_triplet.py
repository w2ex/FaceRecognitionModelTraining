#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:10:41 2019

@author: tpetit
"""

# import libraries
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from myDataloader_preproc import Dataset
from myDataloader_preproc import BalancedBatchSampler
import argparse
import numpy as np
import triplet_selector
import triplet_loss
from tqdm import tqdm
import cv2
import os
cwd = os.getcwd()

import random as rand
from sklearn.metrics import roc_curve

# get arguments

parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str, help = 'Path to dataset.')
parser.add_argument('resultpath', type=str, help = 'Name of the file to write results.')
parser.add_argument('testpath', type=str, help = 'Path to lfw dataset.')
parser.add_argument('-m', '--modelpath', type=str, help = 'Name of the model to save.')
parser.add_argument('-l', '--loadpath', type=str, help = 'Name of the model to load.')
parser.add_argument('-a', '--architecture', type=str, default = 'rn18', help = 'Type of network to train : rn18, rn34, inception_fn, inception_v1')
parser.add_argument('-e', '--epoch', type=int, default = 30, help = 'Max number of epochs.')
parser.add_argument('-c', '--classes', type=int, default = 2, help = 'Number of classes to sample for each batch.')
parser.add_argument('-s', '--samples', type=int, default = 50, help = 'Number of samples per class for each batch.')
parser.add_argument('-d', '--dim', type=int, default = 64, help = 'Embedding dimension.')
parser.add_argument('-g', '--gpu', type=int, default = 0, help = 'GPU number.')
parser.add_argument('-p', '--pairs', type=str, default = 'lfw_pairs.txt', help = 'LFW pairs')

args = parser.parse_args()

datapath = args.datapath

save = False
if args.modelpath is not None :
    modelpath = os.path.join(cwd, args.modelpath)
    save = True
    
load = False
if args.loadpath is not None :
    loadpath = os.path.join(cwd, args.loadpath)
    load = True

writepath = os.path.join(cwd, args.resultpath)
  
max_epochs = args.epoch
n_classes = args.classes
n_samples = args.samples
dim = args.dim
architecture = args.architecture
#datapath = '/home/tpetit/Documents/Datasets/umdfaces_batch3/'
#save = False
#load = False
#max_epochs = 3
#n_classes = 2
#n_samples = 5
#dim = 32

file = open(writepath, 'a')
file.write('Num classes : %d ; num samples per classes : %d ; embbedings dimension : %d \n' % (n_classes, n_samples, dim))
file.close()


# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:"+str(args.gpu) if use_cuda else "cpu")
if use_cuda :
   torch.cuda.set_device(device)
print('The training will take place on the following device : '+str(device))
#cudnn.benchmark = True

# Parameters
params = {'num_workers': 32, 'pin_memory' : True}
device_ids = [0, 1, 2, 3, 4, 5, 6]
# Datasets
labels_train = {}
labels_test = {}
partition = {'train' : [], 'val' : []}

#os.chdir(os.path.join(datapath,'test'))
#classes = os.listdir()
#for i in range(len(classes)//4):
#    os.chdir(os.path.join(datapath,'test', classes[i]))
#    images = os.listdir()
#    rand.shuffle(images)
#    for j in range(5) :
#        labels_test[os.path.join('test', classes[i],images[j])] = i
#        partition['val'].append(os.path.join('test', classes[i],images[j]))

os.chdir(os.path.join(datapath,'train'))
classes = os.listdir()
for i in range(len(classes)): #len(classes)
    os.chdir(os.path.join(datapath,'train', classes[i]))
    images = os.listdir()
    rand.shuffle(images)
    for j in range(len(images)) :
        labels_train[os.path.join('train', classes[i],images[j])] = i
        partition['train'].append(os.path.join('train', classes[i],images[j]))

file = open(os.path.join(cwd, args.pairs), 'r')
splits = []
n_splits = 10
for split in range(n_splits):
    pairs = []
    for i in range(split * 600, (split + 1) * 600):
        line = file.readline()
        p = line[:-1].split('\t')
        if len(p) == 3:
            pair = (os.path.join(args.testpath, p[0], p[0] + '_' + p[1].zfill(4) + '.jpg'),
                    os.path.join(args.testpath, p[0], p[0] + '_' + p[2].zfill(4) + '.jpg'), 1)
        elif len(p) == 4:
            pair = (os.path.join(args.testpath, p[0], p[0] + '_' + p[1].zfill(4) + '.jpg'),
                    os.path.join(args.testpath, p[2], p[2] + '_' + p[3].zfill(4) + '.jpg'), 0)
        pairs.append(pair)
    splits.append(pairs)
file.close()

train_batch_sampler = BalancedBatchSampler(partition['train'], labels_train, n_classes = n_classes, n_samples = n_samples)
#val_batch_sampler = BalancedBatchSampler(partition['val'], labels, n_classes = n_classes, n_samples = n_samples)

# Generators
print('Generating training set...')
training_set = Dataset(partition['train'], labels_train, datapath)
print('Length training set : ', len(training_set))
print('Initializing training set loader...')
training_generator = data.DataLoader(training_set, batch_sampler = train_batch_sampler, **params)
print('Done.')

#print('Generating validation set...')
#validation_set = Dataset(partition['val'], labels_test, datapath)
#print('Length validation set : ', len(validation_set))
#print('Initializing validation set loader...')
#validation_generator = data.DataLoader(validation_set, batch_sampler = val_batch_sampler, **params)
#print('Done.')

if architecture == 'rn18' :
    import Resnet
    net = Resnet.resnet18(num_classes = dim).to(device)
    net = nn.DataParallel(net, device_ids = device_ids)
elif architecture == 'rn34' :
    import Resnet
    net = Resnet.resnet34(num_classes = dim).to(device)
    net = nn.DataParallel(net, device_ids = device_ids)
elif architecture == 'inception_fn' :
    import Inception
    net = Inception.Inception_v1_facenet(num_classes = dim).to(device)
    net = nn.DataParallel(net, device_ids = device_ids)
elif architecture == 'inception_v1' :
    import Inception
    net = Inception.Inception_v1(num_classes = dim).to(device)
    net = nn.DataParallel(net, device_ids = device_ids)
elif architecture == 'vgg16' :
    from torchvision.models import vgg
    net = vgg.vgg16(num_classes = dim).to(device)
    net = nn.DataParallel(net, device_ids = device_ids)
elif architecture == 'vgg16_bn' :
    from torchvision.models import vgg
    net = vgg.vgg16_bn(num_classes = dim).to(device)
    net = nn.DataParallel(net, device_ids = device_ids)
elif architecture == 'inception_resnet' :
    import InceptionResnetV2
    net = InceptionResnetV2.inceptionresnetv2(num_classes = dim, pretrained = False).to(device)
    net = nn.DataParallel(net, device_ids = device_ids)
    
if load :
    net_save = torch.load(loadpath)
    net.load_state_dict(net_save)

margin = 1.
learning_rate = 0.00005 #0.0005
criterion = triplet_loss.OnlineTripletLoss(margin, triplet_selector.RandomNegativeTripletSelector(margin), device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 1e-4)

early_stop = 0
best = 0
epoch = 0
labels_set = list(set([labels_train[x] for x in partition['train']]))
means = {label : np.zeros((dim)) for label in labels_set}

while epoch < max_epochs and early_stop <= 20 :
    
    if epoch >= 1 :
        criterion = triplet_loss.OnlineTripletLoss(margin, triplet_selector.SemihardNegativeTripletSelector(margin), device)
        if learning_rate > 0.000001 : # 0.00001
            learning_rate = max(0.000001, learning_rate * 0.9)
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 1e-4)
    #if epoch >= 2 :
    #    criterion = triplet_loss.OnlineTripletLoss(margin, triplet_selector.HardestNegativeTripletSelector(margin), device)
# Training ------------------------------------------------------------------------
    print('\n Epoch %d, learning rate %.6f' % (epoch+1, learning_rate))
    net.train()
    for i, batch in enumerate(tqdm(training_generator, desc='Batch : '), 1) :
        inputs, labels = batch
        labels_batch = list(set(np.array(labels)))
        label_to_indices = {label: np.where([l == label for l in labels])[0]
                            for label in labels_batch}
        # transfer to GPU
        inputs, labels = inputs.to(device, non_blocking = True), labels.to(device, non_blocking = True)
        outputs = net(inputs)
        out = outputs[:, np.random.choice(list(range(128)), size = 64, replace=False)]
        loss, _ = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        for c in range(n_classes) :
            means[labels_batch[c].item()] = np.mean([outputs[j].detach().cpu().numpy() for j in label_to_indices[labels_batch[c]]], 0)
        train_batch_sampler.set_mean(means)

    print('\nEpoch %d end\n' % (epoch + 1))
    
# Evaluation -----------------------------------------------------------------------    
    net.eval()
    with torch.set_grad_enabled(False):
        distances = []
        labels = []
        similarities = []
        for i in tqdm(range(n_splits)):
            distances.append([])
            labels.append([])
            # similarities.append([])
            for j, pair in enumerate(splits[i]):
                img1 = cv2.imread(pair[0])
                img2 = cv2.imread(pair[1])
                label = pair[2]
                img1 = (torch.from_numpy(img1).permute(2, 1, 0).float() -127.5)/ 128
#                img1 = img1 - img1.mean()
#                img1 = img1 / img1.std()
                img2 = (torch.from_numpy(img2).permute(2, 1, 0).float() -127.5)/ 128
#                img2 = img2 - img2.mean()
#                img2 = img2 / img2.std()
                feat1, feat2 = net(torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0).to(device))
                distance = torch.pow((feat1 - feat2),
                                     2).sum().item()
                distances[-1].append(distance)
                labels[-1].append(label)
        dist_roc = [-d[j] for d in [distances[k] for k in range(len(distances))] for j in range(len(d))]
        lab_roc = [l[j] for l in [labels[k] for k in range(len(labels))] for j in range(len(l))]        
        accuracies = []
        thresholds = []
        for i in range(n_splits):
            test_dist = distances[i]
            test_labels = labels[i]
            train_dist = [-d[j] for d in [distances[k] for k in range(len(distances)) if k != i] for j in range(len(d))]
            train_labels = [l[j] for l in [labels[k] for k in range(len(labels)) if k != i] for j in range(len(l))]
            fp_train, tp_train, thresh = roc_curve(train_labels, train_dist)
            thresh = [-d for d in thresh]
            tot = len(train_labels)
            pos = sum(train_labels)
            neg = tot - pos
            acc_train = np.array([(tp_train[i] * pos + neg - fp_train[i] * neg) / tot for i in range(len(thresh))])
            thresholds.append(thresh[np.argmax(acc_train)])
            opt_thresh = thresh[np.argmax(acc_train)]
            correct = 0
            for k, d in enumerate(test_dist):
                if (d < opt_thresh) == test_labels[k]:
                    correct += 1
            accuracies.append(correct / len(test_dist))
        auc = np.mean(accuracies)
        print('Overall accuracy on LFW : %5f +/- %5f' % (auc, np.std(accuracies)))
#        distances = []
#        labels = []
#        for i, (face1, label1) in enumerate(tqdm(validation_set, desc = 'Evaluation score : ')) :
#            face1 = face1.unsqueeze(0).to(device)
#            for j in range(i+1, len(validation_set)) :
#                face2, label2 = validation_set[j]
#                face2 = face2.unsqueeze(0).to(device)
#                output1, output2 = net(face1), net(face2)
#                distance = torch.pow((output1[0]-output2[0]),2).sum().item()
#                distances.append(distance)
#                labels.append(label1 == label2)
#        distances = [-d for d in distances]
#        auc = roc_auc_score(labels, distances)
#        print('ROC area under curve for validation set : %.3f' %(auc))
        file = open(writepath, 'a')
        file.write(str(epoch+1)+' ; '+str(auc)+'\n')
        file.close()
        if auc > best :
            best = auc
            early_stop = 0
            if save :
                torch.save(net.state_dict(), modelpath+'_epoch'+str(epoch))
        else :
            early_stop += 1
        ## print RoC
    epoch +=1
