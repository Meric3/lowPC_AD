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

import torch
import torch.nn as nn
from pathlib import Path
from torch.autograd import Variable

class Solver():
    def __init__(self, args):
        
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        print(torch.rand(2))
        print("777 is {}, {}".format(0.0819, 0.4911))
        
        self.tf_log = args.tf_log
        
        train_x, test_x, test_y = dataset.dataset(train_path = args.train_path, test_path = args.test_path)
        
        train_x_batchfy = preprocessing.batchify(args, train_x, args.batch_size)
        test_x_batchfy = preprocessing.batchify(args, test_x, args.batch_size)
        generate_batchfy = preprocessing.batchify(args, test_x, 1)
        train_generate_batchfy = preprocessing.batchify(args, train_x, 1)
        
        self.train_x_batchfy = train_x_batchfy[:,:,args.selected_dim']]
        self.test_x_batchfy = test_x_batchfy[:,:,args.selected_dim']]
        self.generate_batchfy = generate_batchfy[:,:,args.selected_dim']]
        self.train_generate_batchfy = train_generate_batchfy[:,:,args.selected_dim']]
        self.test_y = test_y
        

        self.args = args
        self.encoder = model.ENCODER(self.args)
        self.encoder.cuda()

        self.decoder = model.DECODER(self.args)
        self.decoder.cuda()

        self.optim_enc   = torch.optim.Adam(self.encoder.parameters(), self.args.lr'])
        self.optim_dec   = torch.optim.Adam(self.decoder.parameters(), self.args.lr'])

        self.loss_fn = nn.MSELoss()    
    
        self.logger = Logger('./tf_logs')
    
        self.base_dir = Path('model_save')
        self.base_dir.mkdir(parents=True,exist_ok=True)      
        
        self.evaluate = args.evaluate']
    
#     def make_dir_name(self, args):
#         return 'modelName:'+args['model_name']+'__cellType:'+args['cell_type'] \
#                 + '__hidSize:' + str(args['hidden_size']) + '__dropout:' + str(args['dropout'])

    def load(self, path):
        try:
            checkpoint = torch.load(Path(self.base_dir, str(args.selected_dim'])))
            checkpoint = checkpoint.with_suffix('.pth')
            start_epoch = checkpoint['epoch']
            self.encoder.load_state_dict(checkpoint['state_dict_enc'])
            self.optim_enc.load_state_dict((checkpoint['optimizer_enc']))
            self.decoder.load_state_dict(checkpoint['state_dict_dec'])
            self.optim_dec.load_state_dict((checkpoint['optimizer_dec']))            
            del checkpoint
            print("=> loaded checkpoint")
        except:
            print("=> Not exist checkpoint")
            pass        

    def fit(self, load):
        if load == True:
            try:
                load()
                checkpoint = torch.load(Path(self.base_dir,'checkpoint').with_suffix('.pth'))
                start_epoch = checkpoint['epoch']
                self.encoder.load_state_dict(checkpoint['state_dict_enc'])
                self.optim_enc.load_state_dict((checkpoint['optimizer_enc']))
                self.decoder.load_state_dict(checkpoint['state_dict_dec'])
                self.optim_dec.load_state_dict((checkpoint['optimizer_dec']))            
                del checkpoint
                print("=> loaded checkpoint")
            except:
                print("=> Not exist checkpoint")
                pass

        total_loss = 0
        max_f1 = 0
        total_length = self.train_x_batchfy.size(0) - 1
        
        
        log = logging.getLogger('snowdeer_log')
        log.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(levelname)s] (%(filename)s:%(lineno)d) > %(message)s')

        fileHandler = logging.FileHandler('./log.txt')

        fileHandler.setFormatter(formatter)
        log.addHandler(fileHandler)
        
        log.critical(str(args['selected_dim']))
        
                                          
        for epoch in range(0, self.args.epoch']):

            self.encoder.train()
            self.decoder.train()
                
            hidden_enc = self.encoder.init_hidden(self.args.batch_size'])

            for batch, i in enumerate(range(0, self.train_x_batchfy.size(0) - 1, self.args.seq_length'])):
                outSeq = []
                inputSeq, targetSeq = preprocessing.get_batch(self.args, self.train_x_batchfy, i)

                if args.seq_length'] != targetSeq.size()[0] :
                    continue
                hidden_enc = self.encoder.repackage_hidden(hidden_enc)
                self.optim_enc.zero_grad()
                self.optim_dec.zero_grad()
                
                Outputseq_enc, hidden_enc = self.encoder.forward(inputSeq, hidden_enc, return_hiddens=True)
                deccoder_input = Variable(torch.zeros(Outputseq_enc.size())).cuda()
                
                deccoder_input[0,:,:] = Outputseq_enc[-1,:,:] # inputSeq[-1,:,:]
                deccoder_input[1:,:,:] = targetSeq[:-1,:,:]
                
                loss_enc = self.loss_fn(Outputseq_enc[-1,:,:].view(self.args.batch_size'], -1), targetSeq[0,:,:].contiguous().view(self.args.batch_size'], -1))
                loss_enc.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.clip'])
                
                self.optim_enc.step()     
                
                Outputseq_enc, hidden_enc = self.decoder.forward(deccoder_input, hidden_enc, return_hiddens=True)
                loss_dec = self.loss_fn(Outputseq_enc.view(args['batch_size'], -1), targetSeq.contiguous().view(args['batch_size'], -1))   
                loss_dec.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args['clip'])
                self.optim_dec.step()
                
                total_loss += loss_enc.item() + loss_dec.item()        

                if batch % 30 == 0 and self.tf_log == True :
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
            self.anomal_score = postprocessing.get_anomalscore_encdec(base_model = self,                                                                   generate_batchfy = self.generate_batchfy, length = 449916,args = self.args)
#             self.anomal_score = postprocessing.get_anomalscore_minmax(base_model = self,  \
#                                                                  generate_batchfy = self.generate_batchfy, length = 449916,args = self.args)
            self.anomal_score = LA.norm(self.anomal_score, axis=1)
            precision, recall, thresholds = metrics.precision_recall_curve(  self.test_y[:449916].cpu().numpy(), self.anomal_score,pos_label =1)
            beta = 1
            f1 = (1+beta**2)*(precision*recall)/((beta**2*precision)+recall)
            f1 = np.nan_to_num(f1)

            max_pre = precision[np.argmax(f1)]
            max_recall = recall[np.argmax(f1)]
            max_f1_tp = f1[np.argmax(f1)]
            print("epoch[{}]\t precision[{}]\t recall[{}]\t f1[{}]".format(epoch, max_pre, max_recall, max_f1_tp))
            log.info("epoch[{}]\t precision[{}]\t recall[{}]\t f1[{}]".format(epoch, max_pre, max_recall, max_f1_tp))
            
            if max_f1_tp > max_f1:
                self.model_dictionary = {'epoch': epoch + 1,
                        'state_dict_enc': self.encoder.state_dict(),
                        'optimizer_enc': self.optim_enc.state_dict(),
                        'state_dict_dec': self.decoder.state_dict(),
                        'optimizer_dec': self.optim_dec.state_dict(),
                        'args':args,
                        'loss':total_loss,
                        'anomal_score' : self.anomal_score 
                        }
                self.save_checkpoint(self.args, self.model_dictionary)                                    
                                    
            
            
            
 
    def save_checkpoint(self, args, state):
        checkpoint = Path(self.base_dir, str(args['selected_dim']))
        checkpoint = checkpoint.with_suffix('.pth')
        torch.save(state, checkpoint)
        

