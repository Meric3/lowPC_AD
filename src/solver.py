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


class Solver():
    def __init__(self, args, cv_list, log ,cv_i):
        
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        np.random.seed(777)
        
        self.attack_list = pd.read_csv(args.attack_list_path, error_bad_lines=False, sep='\t')

        self.tf_log = args.tf_log
        
        train_x, test_x, test_y = dataset.lstm_dataset(train_path = args.train_path, test_path = args.test_path)
          
        self.cv_list = cv_list
        self.cv_i = cv_i
    
        
        train_x_batchfy = preprocessing.batchify(args, train_x, args.batch_size)
        test_x_batchfy = preprocessing.batchify(args, test_x, args.batch_size)
        generate_batchfy = preprocessing.batchify(args, test_x, 1)
        train_generate_batchfy = preprocessing.batchify(args, train_x, 1)
        
        self.train_x_batchfy = train_x_batchfy[:,:,args.selected_dim]
        self.test_x_batchfy = test_x_batchfy[:,:,args.selected_dim]
        self.generate_batchfy = generate_batchfy[:,:,args.selected_dim]
        self.train_generate_batchfy = train_generate_batchfy[:,:,args.selected_dim]
        self.test_y = test_y
        

        self.args = args
        self.encoder = model.ENCODER(self.args)
        self.encoder.cuda()

        self.decoder = model.DECODER(self.args)
        self.decoder.cuda()

        self.optim_enc   = torch.optim.Adam(self.encoder.parameters(), self.args.lr)
        self.optim_dec   = torch.optim.Adam(self.decoder.parameters(), self.args.lr)

        self.loss_fn = nn.MSELoss()    
    
        self.logger = Logger('./tf_logs')
        
        self.base_dir_cv = Path('model_cv', str(self.cv_i))
        self.base_dir_cv.mkdir(parents=True,exist_ok=True)
        
        self.base_dir = Path('model', str(self.cv_i))
        self.base_dir.mkdir(parents=True,exist_ok=True) 
        

        self.log = log

    def load(self, path):
        try:
            print("=> loaded checkpoint")
        except:
            print("=> Not exist checkpoint")
            pass        

    def fit(self, load):
        total_loss = 0
        

        max_f1_cv = 0
        max_f1_whole = 0
        total_length = self.train_x_batchfy.size(0) - 1
        start_time = time.time()
        
        
                                          
        for epoch in range(0, self.args.epoch):
            self.encoder.train()
            self.decoder.train()
                
            hidden_enc = self.encoder.init_hidden(self.args.batch_size)

            for batch, i in enumerate(range(0, self.train_x_batchfy.size(0) - 1, self.args.seq_length)):
                outSeq = []
                inputSeq, targetSeq = preprocessing.get_batch(self.args, self.train_x_batchfy, i)

                if self.args.seq_length != targetSeq.size()[0] :
                    continue
                hidden_enc = self.encoder.repackage_hidden(hidden_enc)
                self.optim_enc.zero_grad()
                self.optim_dec.zero_grad()
                
                Outputseq_enc, hidden_enc = self.encoder.forward(inputSeq, hidden_enc, return_hiddens=True)
                deccoder_input = Variable(torch.zeros(Outputseq_enc.size())).cuda()
                
                deccoder_input[0,:,:] = Outputseq_enc[-1,:,:] # inputSeq[-1,:,:]
                deccoder_input[1:,:,:] = targetSeq[:-1,:,:]
                
                loss_enc = self.loss_fn(Outputseq_enc[-1,:,:].view(self.args.batch_size, -1), targetSeq[0,:,:].contiguous().view(self.args.batch_size, -1))
                loss_enc.backward(retain_graph=True)
                
                
                encoder_norm = sum(p.grad.data.abs().sum() for p in self.encoder.parameters())
                
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.clip)
                
                self.optim_enc.step()     
                
                Outputseq_enc, hidden_enc = self.decoder.forward(deccoder_input, hidden_enc, return_hiddens=True)
                loss_dec = self.loss_fn(Outputseq_enc.view(self.args.batch_size, -1), targetSeq.contiguous().view(self.args.batch_size, -1))   
                loss_dec.backward()
                
                edecoder_norm = sum(p.grad.data.abs().sum() for p in self.decoder.parameters())
                
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args.clip)
                self.optim_dec.step()
                
                
                
                total_loss += loss_enc.item() + loss_dec.item()


                if batch % 30 == 0 and self.tf_log == True :
                    print(encoder_norm)
                    print(decoder_norm)
                    # 1. Log scalar values (scalar summary)
                    info = { 'enc_loss': loss_enc.item(), 'dec_loss' : loss_dec.item() }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, epoch*total_length + i +1)

                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in self.encoder.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch*total_length + i +1)
                        
                    for tag, value in self.decoder.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch*total_length + i +1)
            
            total_loss = 0    
        

            val_set = self.cv_list[1]
            test_set = self.cv_list[0]

            whole_set = np.arange(self.test_y.shape[0])
                
            if len(self.args.selected_dim) == 1:
                self.anomal_score = postprocessing.get_anomalscore_encdec_1dim(base_model = self,                                                                   generate_batchfy = self.generate_batchfy,args = self.args, cv_list = val_set)
            else:    
                self.anomal_score_cv = postprocessing.get_anomalscore_encdec(self ,self.generate_batchfy, self.args, val_set)
                self.anomal_score_whole = postprocessing.get_anomalscore_encdec(self ,self.generate_batchfy, self.args, whole_set)

            self.anomal_score_cv = LA.norm(self.anomal_score_cv, axis=1)
            self.anomal_score_whole = LA.norm(self.anomal_score_whole, axis=1)

            
            max_conv_tp_cv, max_pre_cv, max_recall_cv, max_f1_tp_cv, \
            max_threshold_tp_cv, find_attack_list_cv, whole_arrack_list_num_cv\
            = postprocessing.evaluate_conv(self.anomal_score_cv, self.test_y, self.attack_list, 400, val_set, self.args)
            
            max_conv_tp_whole, max_pre_whole, max_recall_whole, max_f1_tp_whole,\
            max_threshold_tp_whole, find_attack_list_whole, whole_arrack_list_num_whole\
            = postprocessing.evaluate_conv(self.anomal_score_whole, self.test_y, self.attack_list, 400, whole_set, self.args)

            end_time = time.time()
            if max_f1_tp_whole > max_f1_whole:
#                 print("홀 진입")
                self.log.info("Whole : pr[{}], re[{}], f1[{}], find[{}/{}] ".\
                      format(max_pre_whole, max_recall_whole, max_f1_tp_whole,find_attack_list_whole,whole_arrack_list_num_whole)) 
#                 print("Whole : pr[{}], re[{}], f1[{}], find[{}/{}] ".\
#                       format(max_pre_whole, max_recall_whole, max_f1_tp_whole,find_attack_list_whole,whole_arrack_list_num_whole))
                max_f1_whole = max_f1_tp_whole
            
            print("max_f1_tp_cv 랑 max_F1_cv랑 차이 {}, {}".format(max_f1_tp_cv, max_f1_cv))
            if max_f1_tp_cv > max_f1_cv:
                print("cv 진입")
        
                self.anomal_score_ensemble = postprocessing.get_anomalscore_encdec(self ,self.generate_batchfy, self.args, test_set)
                self.anomal_score_ensemble = LA.norm(self.anomal_score_ensemble, axis=1)

                if max_conv_tp_cv != 0:
                    gauss_kernel = Gaussian1DKernel(max_conv_tp_cv)
                    norm = convolve(self.anomal_score_ensemble, gauss_kernel)
                else:
                    norm = self.anomal_score_ensemble

                self.anomal_score_ensemble  = (norm > max_threshold_tp_cv)
                self.anomal_score_ensemble = self.anomal_score_ensemble.astype(np.int16)
                
                target = self.test_y[test_set]
                if len(target) != len(self.anomal_score_ensemble):
                    self.anomal_score_ensemble = np.vstack((self.anomal_score_ensemble.reshape(-1,1),                                                                 self.anomal_score_ensemble[-3:-1].reshape(-1,1))).reshape(-1)

                attack_list = []
                for i in range(1, len(target)):
                    if target[i] != target[i-1]:
                        attack_list.append(i)
                whole_arrack_list_num = int(len(attack_list)/2)                    

                attack_list_acc = []
                for idx in range(0, len(attack_list),2):
                    k, t = np.unique(self.anomal_score_ensemble[attack_list[idx]:attack_list[idx+1]], return_counts=True)
                    if t.shape[0] > 1:
                        attack_list_acc.append(t[1]/(t[0] + t[1]))
                    elif k == 1:
                        attack_list_acc.append(1.0)
                    elif k == 0:
                        attack_list_acc.append(0.0)
                    else:
                        print("faral error in evaluation")   

                pr = metrics.precision_score(target, self.anomal_score_ensemble)
                re = metrics.recall_score(target, self.anomal_score_ensemble)

                beta = 1
                f1 = (1+beta**2)*(pr*re)/((beta**2*pr)+re)
                f1 = np.nan_to_num(f1)

                find_attack = 0
                for k in range(len(attack_list_acc)):
                    if attack_list_acc[k] != 0.0:
                        find_attack = find_attack + 1        

                self.model_dictionary = {'args':self.args,
                                         'epoch': epoch,
                        'anomal_score_ensemble' : self.anomal_score_ensemble ,
                         'max_conv_tp_cv' : max_conv_tp_cv,
                        'f1' : max_f1_tp_cv,
                        'whole_arrack_list_num' : whole_arrack_list_num,
                                         'cv_list': self.cv_list
                        }
                    
                self.save_checkpoint(self.args, self.model_dictionary, True)
                max_f1_cv = max_f1_tp_cv
                max_conv_cv = max_conv_tp_cv
                max_threshold_cv = max_threshold_tp_cv
                max_pr_cv = max_pre_cv
                max_re_cv = max_recall_cv
                max_find_attack_cv =find_attack_list_cv
                max_whole_arrack_list_num_cv = whole_arrack_list_num_cv
                
#                 최종 cv 출력
#         print("최종")
#         self.log.info("Ensemble : pr[{}], re[{}], f1[{}], find[{}/{}] ".\
#                       format(max_pr_cv, max_re_cv, max_f1_cv,max_find_attack_cv,max_whole_arrack_list_num_cv))
#         print("Ensemble : pr[{}], re[{}], f1[{}], find[{}/{}] ".\
#                       format(max_pr_cv, max_re_cv, max_f1_cv,max_find_attack_cv,max_whole_arrack_list_num_cv))
 
    def save_checkpoint(self, args, state, cv):
        if cv == True:
            checkpoint = Path(self.base_dir_cv, str(args.selected_dim))
            checkpoint = checkpoint.with_suffix('.pth')
        else:
            checkpoint = Path(self.base_dir, str(args.selected_dim))
            checkpoint = checkpoint.with_suffix('.pth')            
        torch.save(state, checkpoint)
        
class Solver_old():
    def __init__(self, args, cv_list, log ,cv_i):
        
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        np.random.seed(777)
        
        self.attack_list = pd.read_csv(args.attack_list_path, error_bad_lines=False, sep='\t')

        self.tf_log = args.tf_log
        
        train_x, test_x, test_y = dataset.lstm_dataset(train_path = args.train_path, test_path = args.test_path)
          
        self.cv_list = cv_list
        self.cv_i = cv_i
    
        
        train_x_batchfy = preprocessing.batchify(args, train_x, args.batch_size)
        test_x_batchfy = preprocessing.batchify(args, test_x, args.batch_size)
        generate_batchfy = preprocessing.batchify(args, test_x, 1)
        train_generate_batchfy = preprocessing.batchify(args, train_x, 1)
        
        self.train_x_batchfy = train_x_batchfy[:,:,args.selected_dim]
        self.test_x_batchfy = test_x_batchfy[:,:,args.selected_dim]
        self.generate_batchfy = generate_batchfy[:,:,args.selected_dim]
        self.train_generate_batchfy = train_generate_batchfy[:,:,args.selected_dim]
        self.test_y = test_y
        

        self.args = args
        self.encoder = model.ENCODER(self.args)
        self.encoder.cuda()

        self.decoder = model.DECODER(self.args)
        self.decoder.cuda()

        self.optim_enc   = torch.optim.Adam(self.encoder.parameters(), self.args.lr)
        self.optim_dec   = torch.optim.Adam(self.decoder.parameters(), self.args.lr)

        self.loss_fn = nn.MSELoss()    
    
        self.logger = Logger('./tf_logs')
        
        self.base_dir_cv = Path('model_cv', str(self.cv_i))
        self.base_dir_cv.mkdir(parents=True,exist_ok=True)
        
        self.base_dir = Path('model', str(self.cv_i))
        self.base_dir.mkdir(parents=True,exist_ok=True) 
        
#         self.evaluate = args.evaluate
    
#     def make_dir_name(self, args):
#         return 'modelName:'+args.model_name+'__cellType:'+args.cell_type \
#                 + '__hidSize:' + str(args.hidden_size) + '__dropout:' + str(args.dropout)



#TODO 로그 정리
        self.log = log
#         self.log.setLevel(logging.DEBUG)

#         self.formatter = logging.Formatter('%(asctime)s > %(message)s')
        
#         now = datetime.now()
        
#         today = '%s-%s-%s'%(now.year, now.month, now.day)
#         second = '-%s-%s-%s'%(now.hour, now.minute, now.second)
        
#         self.fileHandler = logging.FileHandler('./log/' + today + '.txt')

#         self.fileHandler.setFormatter(self.formatter)
#         self.log.addHandler(self.fileHandler)
        
#         self.log.critical("\n\n Now : " + str(today + second))
#         self.log.critical("Explain : " + str(args.explain))
        
#         self.log.critical("Selected dim : " + str(args.selected_dim))

    def load(self, path):
        try:
#             TODO 
#             checkpoint = torch.load(Path(self.base_dir, str(args.selected_dim)))
#             checkpoint = checkpoint.with_suffix('.pth')
#             start_epoch = checkpoint['epoch
#             self.encoder.load_state_dict(checkpoint['state_dict_enc)
#             self.optim_enc.load_state_dict((checkpoint['optimizer_enc))
#             self.decoder.load_state_dict(checkpoint['state_dict_dec)
#             self.optim_dec.load_state_dict((checkpoint['optimizer_dec))            
#             del checkpoint
            print("=> loaded checkpoint")
        except:
            print("=> Not exist checkpoint")
            pass        

    def fit(self, load):
        total_loss = 0
        

        max_f1_cv = 0
        max_f1_whole = 0
        total_length = self.train_x_batchfy.size(0) - 1
        start_time = time.time()
        
        
                                          
        for epoch in range(0, self.args.epoch):

            self.encoder.train()
            self.decoder.train()
                
            hidden_enc = self.encoder.init_hidden(self.args.batch_size)

            for batch, i in enumerate(range(0, self.train_x_batchfy.size(0) - 1, self.args.seq_length)):
                outSeq = []
                inputSeq, targetSeq = preprocessing.get_batch(self.args, self.train_x_batchfy, i)

                if self.args.seq_length != targetSeq.size()[0] :
                    continue
                hidden_enc = self.encoder.repackage_hidden(hidden_enc)
                self.optim_enc.zero_grad()
                self.optim_dec.zero_grad()
                
                Outputseq_enc, hidden_enc = self.encoder.forward(inputSeq, hidden_enc, return_hiddens=True)
                deccoder_input = Variable(torch.zeros(Outputseq_enc.size())).cuda()
                
                deccoder_input[0,:,:] = Outputseq_enc[-1,:,:] # inputSeq[-1,:,:]
                deccoder_input[1:,:,:] = targetSeq[:-1,:,:]
                
                loss_enc = self.loss_fn(Outputseq_enc[-1,:,:].view(self.args.batch_size, -1), targetSeq[0,:,:].contiguous().view(self.args.batch_size, -1))
                loss_enc.backward(retain_graph=True)
                
                
                encoder_norm = sum(p.grad.data.abs().sum() for p in self.encoder.parameters())
                
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.clip)
                
                self.optim_enc.step()     
                
                Outputseq_enc, hidden_enc = self.decoder.forward(deccoder_input, hidden_enc, return_hiddens=True)
                loss_dec = self.loss_fn(Outputseq_enc.view(self.args.batch_size, -1), targetSeq.contiguous().view(self.args.batch_size, -1))   
                loss_dec.backward()
                
                edecoder_norm = sum(p.grad.data.abs().sum() for p in self.decoder.parameters())
                
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args.clip)
                self.optim_dec.step()
                
                
                
                total_loss += loss_enc.item() + loss_dec.item()


                if batch % 30 == 0 and self.tf_log == True :
                    print(encoder_norm)
                    print(decoder_norm)
                    # 1. Log scalar values (scalar summary)
                    info = { 'enc_loss': loss_enc.item(), 'dec_loss' : loss_dec.item() }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, epoch*total_length + i +1)

                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in self.encoder.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch*total_length + i +1)
                        
                    for tag, value in self.decoder.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch*total_length + i +1)
            
            total_loss = 0    
        

            val_set = self.cv_list[1]
            test_set = self.cv_list[0]

            whole_set = np.arange(self.test_y.shape[0])
                
            if len(self.args.selected_dim) == 1:
                self.anomal_score = postprocessing.get_anomalscore_encdec_1dim(base_model = self,                                                                   generate_batchfy = self.generate_batchfy,args = self.args, cv_list = val_set)
            else:    
                self.anomal_score_cv = postprocessing.get_anomalscore_encdec(self ,self.generate_batchfy, self.args, val_set)
                self.anomal_score_whole = postprocessing.get_anomalscore_encdec(self ,self.generate_batchfy, self.args, whole_set)

            self.anomal_score_cv = LA.norm(self.anomal_score_cv, axis=1)
            self.anomal_score_whole = LA.norm(self.anomal_score_whole, axis=1)

            
            max_conv_tp_cv, max_pre_cv, max_recall_cv, max_f1_tp_cv, \
            max_threshold_tp_cv, find_attack_list_cv, whole_arrack_list_num_cv\
            = postprocessing.evaluate_conv(self.anomal_score_cv, self.test_y, self.attack_list, 400, val_set, self.args)
            
            max_conv_tp_whole, max_pre_whole, max_recall_whole, max_f1_tp_whole,\
            max_threshold_tp_whole, find_attack_list_whole, whole_arrack_list_num_whole\
            = postprocessing.evaluate_conv(self.anomal_score_whole, self.test_y, self.attack_list, 400, whole_set, self.args)

            end_time = time.time()
            if max_f1_tp_whole > max_f1_whole:
                self.log.info("Whole : pr[{}], re[{}], f1[{}], find[{}/{}] ".\
                      format(max_pre_whole, max_recall_whole, max_f1_tp_whole,find_attack_list_whole,whole_arrack_list_num_whole))  
                max_f1_whole = max_f1_tp_whole
            
            
            if max_f1_tp_cv > max_f1_cv:
        
                self.anomal_score_ensemble = postprocessing.get_anomalscore_encdec(self ,self.generate_batchfy, self.args, test_set)
                self.anomal_score_ensemble = LA.norm(self.anomal_score_ensemble, axis=1)

                if max_conv_tp_cv != 0:
                    gauss_kernel = Gaussian1DKernel(max_conv_tp_cv)
                    norm = convolve(self.anomal_score_ensemble, gauss_kernel)
                else:
                    norm = self.anomal_score_ensemble

                self.anomal_score_ensemble  = (norm > max_threshold_tp_cv)

                target = self.test_y[test_set]
                if len(target) != len(self.anomal_score_ensemble):
                    self.anomal_score_ensemble = np.vstack((self.anomal_score_ensemble.reshape(-1,1),                                                                 self.anomal_score_ensemble[-3:-1].reshape(-1,1))).reshape(-1)

                attack_list = []
                for i in range(1, len(target)):
                    if target[i] != target[i-1]:
                        attack_list.append(i)
                whole_arrack_list_num = int(len(attack_list)/2)                    

                attack_list_acc = []
                for idx in range(0, len(attack_list),2):
                    k, t = np.unique(self.anomal_score_ensemble[attack_list[idx]:attack_list[idx+1]], return_counts=True)
                    if t.shape[0] > 1:
                        attack_list_acc.append(t[1]/(t[0] + t[1]))
                    elif k == 1:
                        attack_list_acc.append(1.0)
                    elif k == 0:
                        attack_list_acc.append(0.0)
                    else:
                        print("faral error in evaluation")   

                pr = metrics.precision_score(target, self.anomal_score_ensemble)
                re = metrics.recall_score(target, self.anomal_score_ensemble)

                beta = 1
                f1 = (1+beta**2)*(pr*re)/((beta**2*pr)+re)
                f1 = np.nan_to_num(f1)

                find_attack = 0
                for k in range(len(attack_list_acc)):
                    if attack_list_acc[k] != 0.0:
                        find_attack = find_attack + 1        

                self.model_dictionary = {'args':self.args,
                                         'epoch': epoch,
                        'anomal_score_ensemble' : self.anomal_score_ensemble ,
                         'max_conv_tp_cv' : max_conv_tp_cv,
                        'f1' : max_f1_tp_cv,
                        'whole_arrack_list_num' : whole_arrack_list_num
                        }
                    
                self.save_checkpoint(self.args, self.model_dictionary, True)
                max_f1_cv = max_f1_tp_cv
                max_conv_cv = max_conv_tp_cv
                max_threshold_cv = max_threshold_tp_cv
                max_pr_cv = max_pre_cv
                max_re_cv = max_recall_cv
                max_find_attack_cv =find_attack_list_cv
                max_whole_arrack_list_num_cv = whole_arrack_list_num_cv
                
#                 최종 cv 출력
        self.log.info("Ensemble : pr[{}], re[{}], f1[{}], find[{}/{}] ".\
                      format(max_pr_cv, max_re_cv, max_f1_cv,max_find_attack_cv,max_whole_arrack_list_num_cv))          
 
    def save_checkpoint(self, args, state, cv):
        if cv == True:
            checkpoint = Path(self.base_dir_cv, str(args.selected_dim))
            checkpoint = checkpoint.with_suffix('.pth')
        else:
            checkpoint = Path(self.base_dir, str(args.selected_dim))
            checkpoint = checkpoint.with_suffix('.pth')            
        torch.save(state, checkpoint)