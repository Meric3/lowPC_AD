import torch
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
from numpy import linalg as LA
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve

def get_anomalscore_encdec(base_model, generate_batchfy, args, cv_list):
    
    generate_batchfy = generate_batchfy[cv_list,:,:]
   
    outSeq = []
    scores = []
    base_model.encoder.eval()
    base_model.decoder.eval()
    hidden_enc = base_model.encoder.init_hidden(1)
    endPoint = generate_batchfy.shape[0]
    step = args.seq_length
    feature_dim = generate_batchfy.size(-1)

    for i in range(0,endPoint,step):
        Outputseq_enc, hidden_enc = base_model.encoder.forward(Variable(generate_batchfy[i:i+step]).cuda(), hidden_enc)
        deccoder_input = Variable(torch.zeros(Outputseq_enc.size())).cuda()
        deccoder_input[0,:,:] = Outputseq_enc[-1,:,:]
        
        try:
            deccoder_input[1:,:,:] = generate_batchfy[i+step:i+step+step-1,:,:]
        except:
            continue
            
        Outputseq_enc, hidden_enc = base_model.decoder.forward(deccoder_input, hidden_enc, return_hiddens=True)

        error = torch.abs(torch.add( Outputseq_enc.view(-1,generate_batchfy.size(-1)).cpu(), \
                                -generate_batchfy[i:i+step].view(-1,generate_batchfy.size(-1)).cpu()))       
        
        outSeq.append(error)
        
    outSeq = torch.cat(outSeq,dim=0)
    mean = outSeq.mean(dim=0)
    conv_inv = torch.inverse(torch.from_numpy(np.cov((outSeq-mean).t().detach().numpy()))).type(torch.FloatTensor)
    
    for i in range(0,outSeq.shape[0]):
        score = torch.mm(torch.add(outSeq[i], -mean.view(feature_dim,-1).t()),  torch.mm(conv_inv,  torch.add(outSeq[i], -mean.view(feature_dim,-1))))
        scores.append(score.view(-1,generate_batchfy.size(-1)))
        
    scores = torch.cat(scores,dim=0)
    scores = scores.detach().numpy()
    nomalize_scores =  (scores - scores.mean(axis=0))/scores.var(axis=0)
    return nomalize_scores



def get_anomalscore_minmax(base_model, generate_batchfy, length,args):
    outSeq = []
    scores = []
    base_model.encoder.eval()
    base_model.decoder.eval()
    hidden_enc = base_model.encoder.init_hidden(1)
    endPoint = length #446000 
    step = args.seq_length
    feature_dim = generate_batchfy.size(-1)

    for i in range(0,endPoint,step):
        Outputseq_enc, hidden_enc = base_model.encoder.forward(Variable(generate_batchfy[i:i+step]).cuda(), hidden_enc)
        deccoder_input = Variable(torch.zeros(Outputseq_enc.size())).cuda()
        deccoder_input[0,:,:] = Outputseq_enc[-1,:,:]
        try:
            deccoder_input[1:,:,:] = generate_batchfy[i+step:i+step+step-1,:,:]
        except:
            continue
        Outputseq_enc, hidden_enc = base_model.decoder.forward(deccoder_input, hidden_enc, return_hiddens=True)
        if Outputseq_enc.size()[0] != generate_batchfy[i:i+step].size()[0] :
            continue
        error = torch.abs(torch.add( Outputseq_enc.view(-1,generate_batchfy.size(-1)).cpu(), \
                                -generate_batchfy[i:i+step].view(-1,generate_batchfy.size(-1)).cpu()))       
        outSeq.append(error)
        
    outSeq = torch.cat(outSeq,dim=0)
    outSeq = outSeq.detach().numpy()
    return outSeq - outSeq.min(axis=0) / (outSeq.max(axis=0) - outSeq.min(axis=0) )
#     return outSeq - outSeq.min(dim=0)[0] / (outSeq.max(dim=0)[0]  - outSeq.min(dim=0)[0] )

def get_anomalscore_encdec_1dim(base_model, generate_batchfy, length,args, cv_list):
   
    outSeq = []
    scores = []
    base_model.encoder.eval()
    base_model.decoder.eval()
    hidden_enc = base_model.encoder.init_hidden(1)
    endPoint = length #446000 
    step = args.seq_length
    feature_dim = generate_batchfy.size(-1)

    for i in range(0,endPoint,step):
        Outputseq_enc, hidden_enc = base_model.encoder.forward(Variable(generate_batchfy[i:i+step]).cuda(), hidden_enc)
        deccoder_input = Variable(torch.zeros(Outputseq_enc.size())).cuda()
        deccoder_input[0,:,:] = Outputseq_enc[-1,:,:]
        try:
            deccoder_input[1:,:,:] = generate_batchfy[i+step:i+step+step-1,:,:]
        except:
            continue
        Outputseq_enc, hidden_enc = base_model.decoder.forward(deccoder_input, hidden_enc, return_hiddens=True)
        if Outputseq_enc.size()[0] != generate_batchfy[i:i+step].size()[0] :
            continue
        error = torch.abs(torch.add( Outputseq_enc.view(-1,generate_batchfy.size(-1)).cpu(), \
                                -generate_batchfy[i:i+step].view(-1,generate_batchfy.size(-1)).cpu()))       
        outSeq.append(error)
        
    outSeq = torch.cat(outSeq,dim=0)
    mean = outSeq.mean(dim=0)
    var = outSeq.std(dim=0)**2
#     print("anomal_detection...")
    
    scores = (outSeq - mean)*(outSeq - mean)/(var)
    
    scores = scores.detach().numpy()
    nomalize_scores =  (scores - scores.mean(axis=0))/scores.var(axis=0)
    return nomalize_scores



def evaluate_conv(anomal_score, test_y, attack_list, conv, cv_list, args):
    max_f1 = 0

    for conv in range(0, conv, 10):
        if conv != 0:
            gauss_kernel = Gaussian1DKernel(conv)
            norm = convolve(anomal_score, gauss_kernel)
        else:
            norm = anomal_score
        target = test_y[cv_list].cpu().numpy()

        if target.shape[0] != norm.shape[0]:
            target = target[:-args.seq_length]

        precision, recall, thresholds = metrics.precision_recall_curve(target, norm ,pos_label =1)

        beta = 1
        f1 = (1+beta**2)*(precision*recall)/((beta**2*precision)+recall)
        f1 = np.nan_to_num(f1)

        max_pre_tp = precision[np.argmax(f1)]
        max_recall_tp = recall[np.argmax(f1)]
        max_f1_tp = f1[np.argmax(f1)]

        attack_list_acc = []
        anomaly = (norm > thresholds[np.argmax(f1)] )
        anomaly = anomaly.astype(np.int16)
        if args.cross_validation == True:
            attack_list = []
            for i in range(1, len(target)):
                if target[i] != target[i-1]:
                    attack_list.append(i)
            whole_arrack_list_num = len(attack_list)/2
            for idxx in range(0, len(attack_list),2):
                k, t = np.unique(anomaly[attack_list[idxx]:attack_list[idxx+1]], return_counts=True)
                if t.shape[0] > 1:
                    attack_list_acc.append(t[1]/(t[0] + t[1]))
                elif k == 1:
                    attack_list_acc.append(1.0)
                elif k == 0:
                    attack_list_acc.append(0.0)
                else:
                    print("faral error in evaluation")
                    print(k)
                    print(t)
                 
        else:                
            whole_arrack_list_num = len(attack_list)
            for idxx in range(0, len(attack_list)):
                k, t = np.unique(anomaly[attack_list['Start Time'][idxx]:attack_list['End Time'][idxx]], return_counts=True)
                if t.shape[0] > 1:
                    attack_list_acc.append(t[1]/(t[0] + t[1]))
                elif k == 1:
                    attack_list_acc.append(1.0)
                elif k == 0:
                    attack_list_acc.append(0.0)
                else:
                    print("faral error in evaluation")
                
        find_attack_list = []
        for k in range(len(attack_list_acc)):
            if attack_list_acc[k] != 0.0:
                find_attack_list.append(k)               

        if max_f1 < max_f1_tp:
            max_pre = max_pre_tp
            max_recall = max_recall_tp
            max_f1 = max_f1_tp
            max_conv = conv
            max_find_attack_list = find_attack_list
            max_threshold = thresholds[np.argmax(f1)]
        

#     print("conv[{}]\t precision[{}]\t recall[{}]\t f1[{}] \t find attack [{}]".format(max_conv, max_pre, max_recall, max_f1))
    return max_conv, max_pre, max_recall, max_f1, max_threshold,  max_find_attack_list, whole_arrack_list_num