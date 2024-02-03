
# coding: utf-8
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
from sklearn import metrics

from astropy.convolution import Gaussian1DKernel, convolve

import torch
import torch.nn as nn
from pathlib import Path
from torch.autograd import Variable

import time
from datetime import datetime 

from sklearn.model_selection import KFold
import sklearn.preprocessing 

import itertools as it
import utils.postprocessing as postprocessing

import solver as Solver

def main():       
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
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--cell_type", type=str, default="LSTM")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=2)
    parser.add_argument('--selected_dim', nargs='+', type=int, default=[36, 38, 28, 40])

    parser.add_argument("--cross_validation", default=False, action="store_true")
    parser.add_argument("--explain", type=str, default="cross_validation")
    parser.add_argument("--cv_start", type=int, default=0)
    parser.add_argument("--cv_end", type=int, default=5)
    args = parser.parse_args()


    log = logging.getLogger('LSTM_log')
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    now = datetime.now()

    today = '%s-%s-%s'%(now.year, now.month, now.day)
    second = '-%s-%s-%s'%(now.hour, now.minute, now.second)

    fileHandler = logging.FileHandler('./log/' + today + "-"+args.explain + '.txt')

    fileHandler.setFormatter(formatter)
    log.addHandler(fileHandler)
    log.info("-"*99)
    log.info("Today {} ".format(today + second))

    _, test_x, _ = dataset.lstm_dataset(train_path = args.train_path, test_path = args.test_path)

    

    cv_list = []
    cv = KFold(5, shuffle=False, random_state=0)
    for i, (idx_train, idx_test) in enumerate(cv.split(test_x)):
        cv_list.append([idx_train, idx_test])

    all_pcs = [0,1,2,38,39, 40]
    combi_ = list(it.combinations(all_pcs, 3))


    cv_list_select = [ 1, 4]
    for cv_tp in range(2):
        cv_i = cv_list_select[cv_tp]
        anomal_score_ensemble_list = []
        log.info("\n\n cv_list = {} ".format(cv_i))        
        for cv_combi in range(len(combi_)):
            args.selected_dim = list(combi_[cv_combi])
            log.critical("-"*99)
            log.critical("Selected dim : " + str(args.selected_dim))
            print(args.selected_dim)

            solver = Solver.Solver(args = args, cv_list = cv_list[cv_i], log = log, cv_i = cv_i)
            solver.fit(load = False)
        
if __name__ == "__main__":
    main()
