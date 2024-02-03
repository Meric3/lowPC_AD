import utils.dataset as dataset
import utils.preprocessing as preprocessing
from utils.logger import Logger
import lstm.model as model
import utils.postprocessing as postprocessing
import pandas as pd
from sklearn import metrics
from numpy import linalg as LA
import numpy as np
import logging
import argparse
from sklearn import svm

from astropy.convolution import Gaussian1DKernel, convolve

# import torch
# import torch.nn as nn
from pathlib import Path
from torch.autograd import Variable

import time

import os

import sklearn.preprocessing 
from sklearn.decomposition import PCA

import itertools as it

data_path = '../DATA/wadi_data'
# train_x, test_x, _, test_y= Wadi_dataset(data_path)

# SWaT DATA LOAD Func
def Wadi_dataset(data_path):

    train_path = os.path.join(data_path, 'train')
    var_path = os.path.join(data_path, 'test')    
    test_path = os.path.join(data_path, 'test')    

    train_x = pd.read_csv(train_path + '_x.csv')
    train_y = pd.read_csv(train_path  + '_y.csv')
    test_x = pd.read_csv(test_path + '_x.csv')
    test_y = pd.read_csv(test_path  + '_y.csv')    


    train_y = torch.FloatTensor(train_y.to_numpy()).cuda()
    test_y = torch.FloatTensor(test_y.to_numpy()).cuda()

    unique, counts = np.unique(train_y, return_counts=True)
    print('In train :', dict(zip(unique, counts)))
    unique, counts = np.unique(test_y, return_counts=True)
    print('In test :', dict(zip(unique, counts)))     
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_x)
    # train zero center about train data
    train_x = scaler.transform(train_x)

    # test zero center about train data
    test_x = scaler.transform(test_x)

    pca = PCA(n_components = 93)

    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    train_x = torch.FloatTensor(train_x).cuda()
    test_x = torch.FloatTensor(test_x).cuda()

    train_row = train_x.size(0)
    train_col = train_x.size(1)
    test_row = test_x.size(0) 
    test_col = test_x.size(1)
        
    
    return  train_x, test_x, train_y, test_y
        
    
    
    
train_path = os.path.join(data_path, 'train')
var_path = os.path.join(data_path, 'test')    
test_path = os.path.join(data_path, 'test')    

train_x = pd.read_csv(train_path + '_x.csv')
train_y = pd.read_csv(train_path  + '_y.csv')
test_x = pd.read_csv(test_path + '_x.csv')
test_y = pd.read_csv(test_path  + '_y.csv')    


test_y[test_y == +1] = -1  
test_y[test_y == 0] = +1 
test_y = test_y['label']

def pca_return(x_train, x_test, variance = True):
    if variance == False:
        scaler = sklearn.preprocessing.StandardScaler(with_std = False)
    else:
        scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(x_train)
    
    # train zero center about train data. For fairness test dataset scaled from x_train information
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(n_components = 93)
    pca.fit(x_train)
    
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    
    return x_train, x_test

def check(data) :
    if -1 in data :
        return -1
    else : 
        return 1
    
train_x, test_x = pca_return(train_x, test_x, True)


parser = argparse.ArgumentParser()

parser.add_argument("--cuda", default=True, action="store_true")
parser.add_argument("--tf_log", default=False, action="store_true")
parser.add_argument("--model_name", type=str, default="enc_dec")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--clip", type=int, default=1)

parser.add_argument("--train_path", type=str, default="../../DATA/SWaT/SWaT_Physical/SWaT_Dataset_Normal_v0.csv")
parser.add_argument("--test_path", type=str, default="../../DATA/SWaT/SWaT_Physical/SWaT_Dataset_Attack_v0.csv")
parser.add_argument("--attack_list_path", type=str, default='../../DATA/SWaT/SWaT_Physical/attack_list.csv')

parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden_size", type=int, default=4)
parser.add_argument("--nlayers", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.0001)

parser.add_argument("--cell_type", type=str, default="LSTM")
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--seq_length", type=int, default=2)
parser.add_argument('--selected_dim', nargs='+', type=int, default=[0,1,2])
args = parser.parse_args()

candi = [0,1,2,3,82,83,84,85]
combi_ = list(it.combinations(candi, 3))
for candi in combi_:
    candi = list(candi)
    train_x_ = train_x[:,candi]
    test_x_ = test_x[:, candi]
    
    nu = 0.001
    gamma = 0.001
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma, verbose=False)
    clf.fit(train_x_)
    start_time = time.time()
    preds = clf.predict(test_x_)      
    end_time = time.time()
    f1 = metrics.f1_score(test_y, preds, pos_label = -1)
    print(candi)
    print(end_time-start_time)
    print(f1)