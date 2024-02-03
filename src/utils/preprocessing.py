import torch
from torch.autograd import Variable


def batchify(args,data, batch_size):
    
#     data = torch.FloatTensor(data).cuda()
    
    nbatch = data.size(0) // batch_size
    trimmed_data = data.narrow(0,0,nbatch * batch_size)
    batched_data = trimmed_data.contiguous().view(batch_size, -1, trimmed_data.size(-1)).transpose(0,1)
    if args.cuda:
        batched_data = batched_data.cuda()
    return batched_data

def get_batch(args, source, i, evaluation=False):
    seq_len = min(args.seq_length, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation) # [ seq_len * batch_size * feature_size ]
    try:
        target = Variable(source[i+seq_len:i+seq_len+seq_len]) # [ (seq_len x batch_size x feature_size) ]
    except:
        return data, data
    return data, target   