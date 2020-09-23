from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
from bisect import bisect_right

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
    return adj

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=0.01,
        warmup_iters=20.,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
        #print(self.last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                #print(self.last_epoch)
                alpha = (self.last_epoch + 1) / self.warmup_iters
                #print(alpha)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                #print(warmup_factor)
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

