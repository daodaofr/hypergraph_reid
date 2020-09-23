import torch
import torch.nn as nn
import os, shutil
import numpy as np
import scipy.sparse as sp
from torch.autograd import Variable

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def build_adj(t=4, p=4):
    rows = []
    cols = []
    for j in range(t-1):
        for i in range(p):
            if i == 0:
                rows += [i+j*p, i+j*p]
                cols += [i+(j+1)*p, i+(j+1)*p+1]
            elif i == p-1:
                rows += [i+j*p, i+j*p]
                cols += [i+(j+1)*p-1, i+(j+1)*p]
            else:
                rows += [i+j*p, i+j*p, i+j*p]
                cols += [i+(j+1)*p-1, i+(j+1)*p, i+(j+1)*p+1]
    data = np.ones(len(rows))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(t*p, t*p), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def build_adj_full(t=4, p=4):
    rows = []
    cols = []
    for j in range(t-1):
        for i in range(p):
            rows += [i+j*p for k in range(p)]
            cols += range((j+1)*p, (j+1)*p+p)
    data = np.ones(len(rows))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(t*p, t*p), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def build_adj_full_d(t=4, p=4, d=1):
    rows = []
    cols = []
    for dd in range(d):
        for j in range(t-dd-1):
            for i in range(p):
                rows += [i+j*p for k in range(p)]
                cols += range((j+1+dd)*p, (j+1+dd)*p+p)
    data = np.ones(len(rows))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(t*p, t*p), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def build_adj_full_full(t=4, p=4):
    rows = []
    cols = []
    for j in range(t-1):
        for i in range(p):
            rows += [i+j*p for k in range(p*(t-1-j))]
            cols += range((j+1)*p, p*t)
    data = np.ones(len(rows))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(t*p, t*p), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def build_adj_full_circle(t=4, p=4):
    rows = []
    cols = []
    for j in range(t-1):
        for i in range(p):
            if j == 0:
                rows += [i+j*p for k in range(p)]
                cols += range((t-1)*p, (t-1)*p + p)
            rows += [i+j*p for k in range(p)]
            cols += range((j+1)*p, (j+1)*p+p)

    data = np.ones(len(rows))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(t*p, t*p), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, epoch, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    torch.save({'epoch': epoch+1}, os.path.join(path, 'misc.pt'))


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = embed._backend.Embedding.apply(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return X


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


def mask2d(B, D, keep_prob, cuda=True):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m = Variable(m, requires_grad=False)
    if cuda:
        m = m.cuda()
    return m

