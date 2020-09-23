from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from .convlstm import ConvLSTM
import math
import random
from .resnet import ResNet, BasicBlock, Bottleneck, ResNetNonLocal
from .utils import build_adj_full_full, build_adj_full_d
__all__ = ['ResNet50GRAPHPOOLPARTHyper']

class NearestConvolution(nn.Module):
    """
    Use both neighbors on graph structures and neighbors of nearest distance on embedding space
    """
    def __init__(self, dim_in, dim_out):
        super(NearestConvolution, self).__init__()

        self.kn = 3
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=False)
        self.dropout = nn.Dropout(p=0.1)

        self.trans = ConvMapping(self.dim_in, self.kn)

    def _nearest_select(self, feats):
        b = feats.size()[0]
        N = feats.size()[1]
        dis = NearestConvolution.cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=2)
        #k_nearest = torch.stack([feats[idx[i]] for i in range(N)], dim=0)
        k_nearest = torch.stack([torch.stack([feats[j, idx[j, i]] for i in range(N)], dim=0) for j in range(b)], dim=0)                                        # (b, N, self.kn, d)
        return k_nearest

    @staticmethod
    def cos_dis(X):
        """
        cosine distance
        :param X: (b, N, d)
        :return: (b, N, N)
        """
        X = nn.functional.normalize(X, dim=2, p=2)
        XT = X.transpose(1, 2)                             #(b, d, N)
        return torch.bmm(X, XT)                            #(b, N, N)
        #return torch.matmul(X, XT)

    def forward(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats                                           # (b, N, d)
        x1 = self._nearest_select(x)                        # (b, N, kn, d)
        x_list = []
        for i in range(x1.shape[0]):
            x = self.trans(x1[i])                                  # (N, d)
            x = F.relu(self.fc(self.dropout(x)))       # (N, d')
            x_list.append(x)
        x = torch.stack(x_list, dim=0)                      #(b, N, d')
        return x

class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGraphSAGE, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        if self.use_bn:
            self.bn = nn.BatchNorm1d(outfeat)
            #self.bn = nn.BatchNorm1d(16)

    def forward(self, x, adj):
        #print(adj.shape)
        #print(x.shape)
        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        #print(h_k.shape)
        if self.use_bn:
            #self.bn = nn.BatchNorm1d(h_k.size(1))
            h_k = self.bn(h_k.permute(0,2,1).contiguous())
            #print(h_k.shape)
            h_k = h_k.permute(0, 2, 1)
            #print(h_k.shape)

        return h_k

def sampler_fn(adj):
    n = adj.size(0)
    #print(adj.data)
    adj = adj.data>0
    n_max = adj.sum(dim=0).max() - 1
    nei = []
    for i in range(n):
        tmp = [j for j in range(n) if adj[i,j]>0 and j != i]
        if len(tmp) != n_max:
            tmp += tmp
            random.shuffle(tmp)
            tmp = tmp[0:n_max]
        nei += tmp
    return nei

class BatchedGAT_cat1(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGAT_cat1, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        self.num_head = 1

        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        self.W_a = nn.ModuleList([nn.Linear(2*infeat, 1, bias=False) for i in range(self.num_head)])
        for i in range(self.num_head):
            nn.init.xavier_uniform_(self.W_a[i].weight, gain=nn.init.calculate_gain('relu'))

        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.use_bn:
            self.bn = nn.BatchNorm1d((self.num_head + 1) * outfeat)

    def forward(self, x, adj):
        b = x.size(0)
        h_k_list = []
        #x = self.W_x(x)

        sample_size = adj.size(0)
        assert(sample_size == x.size(1))
        idx_neib = sampler_fn(adj)
        x_neib = x[:, idx_neib, :].contiguous()
        x_neib = x_neib.view(b, sample_size, -1, x_neib.size(2))
        #print(x_neib.shape)

        a_input = torch.cat((x.unsqueeze(2).repeat(1, 1, x_neib.size(2), 1), x_neib), 3)
        #print(a_input.shape)
        h_k = self.W_x(x)
        #h_k_junk = self.W_x(x[i, sample_size:, :].unsqueeze(0))
        for j in range(self.num_head):
            e = self.leakyrelu(self.W_a[j](a_input).squeeze(3))
            #print(e.shape)
            attention = F.softmax(e, dim=2)
            #print(attention.shape)
            h_prime = torch.matmul(attention.unsqueeze(2), x_neib)
            #print(h_k.shape)
            #print(h_prime.shape)
            h_k = torch.cat((h_k, self.W_neib(h_prime.squeeze(2))), 2)
            #h_k_junk = torch.cat((h_k_junk, self.W_neib(x[i, sample_size:, :].unsqueeze(0))), 2)
        #h_k = torch.cat((h_k, h_k_junk), 1)
        h_k_list.append(h_k)
        h_k_f = torch.cat(h_k_list, dim=2)

        h_k_f = F.normalize(h_k_f, dim=2, p=2)
        h_k_f = F.relu(h_k_f)
        if self.use_bn:
            h_k_f = self.bn(h_k_f.permute(0, 2, 1).contiguous())
            h_k_f = h_k_f.permute(0, 2, 1)


        return h_k_f

class BatchedGAT_cat1Temporal(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGAT_cat1Temporal, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        self.num_head = 1

        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        self.W_a = nn.ModuleList([nn.Linear(2*infeat, 1, bias=False) for i in range(self.num_head)])
        for i in range(self.num_head):
            nn.init.xavier_uniform_(self.W_a[i].weight, gain=nn.init.calculate_gain('relu'))

        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.use_bn:
            self.bn = nn.BatchNorm1d((self.num_head*3 + 1) * outfeat)

    def forward(self, x, adj1, adj2, adj3):
        b = x.size(0)
        h_k_list = []
        #x = self.W_x(x)

        sample_size1 = adj1.size(0)
        assert(sample_size1 == x.size(1))
        idx_neib1 = sampler_fn(adj1)
        x_neib1 = x[:, idx_neib1, :].contiguous()
        x_neib1 = x_neib1.view(b, sample_size1, -1, x_neib1.size(2))
        a_input1 = torch.cat((x.unsqueeze(2).repeat(1, 1, x_neib1.size(2), 1), x_neib1), 3)

        sample_size2 = adj2.size(0)
        assert(sample_size2 == x.size(1))
        idx_neib2 = sampler_fn(adj2)
        x_neib2 = x[:, idx_neib2, :].contiguous()
        x_neib2 = x_neib2.view(b, sample_size2, -1, x_neib2.size(2))
        a_input2 = torch.cat((x.unsqueeze(2).repeat(1, 1, x_neib2.size(2), 1), x_neib2), 3)

        sample_size3 = adj3.size(0)
        assert(sample_size3 == x.size(1))
        idx_neib3 = sampler_fn(adj3)
        x_neib3 = x[:, idx_neib3, :].contiguous()
        x_neib3 = x_neib3.view(b, sample_size3, -1, x_neib3.size(2))
        a_input3 = torch.cat((x.unsqueeze(2).repeat(1, 1, x_neib3.size(2), 1), x_neib3), 3)

        h_k = self.W_x(x)
        for j in range(self.num_head):
            e1 = self.leakyrelu(self.W_a[j](a_input1).squeeze(3))
            attention1 = F.softmax(e1, dim=2)
            h_prime1 = torch.matmul(attention1.unsqueeze(2), x_neib1)

            e2 = self.leakyrelu(self.W_a[j](a_input2).squeeze(3))
            attention2 = F.softmax(e2, dim=2)
            h_prime2 = torch.matmul(attention2.unsqueeze(2), x_neib2)

            e3 = self.leakyrelu(self.W_a[j](a_input3).squeeze(3))
            attention3 = F.softmax(e3, dim=2)
            h_prime3 = torch.matmul(attention3.unsqueeze(2), x_neib3)

            h_k = torch.cat((h_k, self.W_neib(h_prime1.squeeze(2)), self.W_neib(h_prime2.squeeze(2)),  self.W_neib(h_prime3.squeeze(2))), 2)
        #h_k_list.append(h_k)

        #h_k_f = torch.cat(h_k_list, dim=2)
        h_k_f = h_k

        h_k_f = F.normalize(h_k_f, dim=2, p=2)
        h_k_f = F.relu(h_k_f)

        if self.use_bn:
            h_k_f = self.bn(h_k_f.permute(0, 2, 1).contiguous())
            h_k_f = h_k_f.permute(0, 2, 1)


        return h_k_f

class BatchedGraphSAGEMean1(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGraphSAGEMean1, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        #print(infeat,outfeat)
        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        if self.use_bn:
            self.bn = nn.BatchNorm1d(2*outfeat)
            #self.bn = nn.BatchNorm1d(16)

    def forward(self, x, adj):
        #print(adj.shape)
        #print(x.shape)
        idx_neib = sampler_fn(adj)
        x_neib = x[:,idx_neib,].contiguous()
        #print(x_neib.shape)

        x_neib = x_neib.view(x.size(0), x.size(1), -1, x_neib.size(2))
        x_neib = x_neib.mean(dim=2)
        #print(x_neib.shape)

        h_k = torch.cat((self.W_x(x), self.W_neib(x_neib)), 2)

        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        #print(h_k.shape)
        if self.use_bn:
            #self.bn = nn.BatchNorm1d(h_k.size(1))
            h_k = self.bn(h_k.permute(0,2,1).contiguous())
            #print(h_k.shape)
            h_k = h_k.permute(0, 2, 1)
            #print(h_k.shape)

        return h_k


class BatchedGraphSAGEDynamicMean1(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGraphSAGEDynamicMean1, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        #print(infeat,outfeat)
        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        if self.use_bn:
            self.bn = nn.BatchNorm1d(2*outfeat)
            #self.bn = nn.BatchNorm1d(16)

        self.kn = 3

    def forward(self, x, adj):
        #print(adj.shape)
        #print(x.shape)

        b = x.size()[0]
        N = x.size()[1]
        dis = NearestConvolution.cos_dis(x)
        #tk = min(dis.shape[2], self.kn)
        _, idx = torch.topk(dis, self.kn, dim=2)
        #k_nearest = torch.stack([feats[idx[i]] for i in range(N)], dim=0)
        k_nearest = torch.stack([torch.stack([x[j, idx[j, i]] for i in range(N)], dim=0) for j in range(b)], dim=0)                                        # (b, N, self.kn, d)

        x_neib = k_nearest[:,:,1:,].contiguous()

        #x_neib = x_neib.view(x.size(0), x.size(1), -1, x_neib.size(2))
        x_neib = x_neib.mean(dim=2)
        #print(k_nearest.shape)
        #x_cmp = x - k_nearest[:,:,0]
        #print(torch.sum(x_cmp)) 

        h_k = torch.cat((self.W_x(x), self.W_neib(x_neib)), 2)

        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        #print(h_k.shape)
        if self.use_bn:
            #self.bn = nn.BatchNorm1d(h_k.size(1))
            h_k = self.bn(h_k.permute(0,2,1).contiguous())
            #print(h_k.shape)
            h_k = h_k.permute(0, 2, 1)
            #print(h_k.shape)

        return h_k

class BatchedGraphSAGEDynamicRangeMean1(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGraphSAGEDynamicRangeMean1, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        #print(infeat,outfeat)
        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        if self.use_bn:
            self.bn = nn.BatchNorm1d(2*outfeat)
            #self.bn = nn.BatchNorm1d(16)

        self.kn = 3

    def forward(self, x, adj, p, t):
        #print(adj.shape)
        #print(x.shape)
        # x: (b, N, d)
        b = x.size()[0]
        N = x.size()[1]
        k_nearest_list = []
        tk = self.kn
        for i in range(int(N/p)):
            idx_start = max(0, i-t)
            idx_end = min(i+t+1, int(N/p))
            tmp_x = x[:,idx_start*p:idx_end*p,]
            dis = NearestConvolution.cos_dis(tmp_x)
            if i==0:
              tk = min(dis.shape[2], self.kn)
              #print(tk)
            _, idx = torch.topk(dis, tk, dim=2)
            #print(tmp_x.shape)
            #print(idx,idx.shape)
            #print(dis,dis.shape)
            k_nearest = torch.stack([torch.stack([tmp_x[j, idx[j, i]] for i in range(p*(idx_end-idx_start))], dim=0) for j in range(b)], dim=0) #(b, x*p, kn, d)
            #print(k_nearest)
            k_nearest_list.append(k_nearest[:,p*(i-idx_start):p*(i-idx_start+1),])
        k_nearest = torch.cat(k_nearest_list, dim=1) #(b,N, kn, d)
        x_neib = k_nearest[:,:,1:,].contiguous()

        #x_neib = x_neib.view(x.size(0), x.size(1), -1, x_neib.size(2))
        x_neib = x_neib.mean(dim=2)
        #print(k_nearest.shape)
        #x_cmp = x - k_nearest[:,:,0]
        #print(torch.sum(x_cmp)) 

        h_k = torch.cat((self.W_x(x), self.W_neib(x_neib)), 2)

        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        #print(h_k.shape)
        if self.use_bn:
            #self.bn = nn.BatchNorm1d(h_k.size(1))
            h_k = self.bn(h_k.permute(0,2,1).contiguous())
            #print(h_k.shape)
            h_k = h_k.permute(0, 2, 1)
            #print(h_k.shape)

        return h_k

class BatchedGraphSAGEMean1Temporal(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGraphSAGEMean1Temporal, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        #print(infeat,outfeat)
        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        if self.use_bn:
            self.bn = nn.BatchNorm1d(4*outfeat)
            #self.bn = nn.BatchNorm1d(16)

    def forward(self, x, adj1, adj2, adj3):
        #print(adj.shape)
        #print(x.shape)
        idx_neib1 = sampler_fn(adj1)
        x_neib1 = x[:,idx_neib1,].contiguous()

        x_neib1 = x_neib1.view(x.size(0), x.size(1), -1, x_neib1.size(2))
        x_neib1 = x_neib1.mean(dim=2)
        #print(x_neib.shape)

        idx_neib2 = sampler_fn(adj2)
        x_neib2 = x[:,idx_neib2,].contiguous()

        x_neib2 = x_neib2.view(x.size(0), x.size(1), -1, x_neib2.size(2))
        x_neib2 = x_neib2.mean(dim=2)

        idx_neib3 = sampler_fn(adj3)
        x_neib3 = x[:,idx_neib3,].contiguous()

        x_neib3 = x_neib3.view(x.size(0), x.size(1), -1, x_neib3.size(2))
        x_neib3 = x_neib3.mean(dim=2)

        h_k = torch.cat((self.W_x(x), self.W_neib(x_neib1), self.W_neib(x_neib2), self.W_neib(x_neib3)), 2)

        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        #print(h_k.shape)
        if self.use_bn:
            #self.bn = nn.BatchNorm1d(h_k.size(1))
            h_k = self.bn(h_k.permute(0,2,1).contiguous())
            #print(h_k.shape)
            h_k = h_k.permute(0, 2, 1)
            #print(h_k.shape)

        return h_k

class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, is_final=False, link_pred=False):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.is_final = is_final
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)
        self.log = {}
        self.link_pred_loss = 0
        self.entropy_loss = 0

    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        if self.link_pred:
            # TODO: Masking padded s_l
            self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
            self.entropy_loss = torch.distributions.Categorical(probs=s_l).entropy()
            self.entropy_loss = self.entropy_loss.sum(-1)
        return xnext, anext



class ResNet50GRAPHPOOLPARTHyper(nn.Module):
    def __init__(self, pool_size, input_shape, n_classes, loss={'xent'}):
        super(ResNet50GRAPHPOOLPARTHyper, self).__init__()
        self.pool_size = pool_size
        self.input_shape = input_shape
        self.link_pred = False
        self.hidden_dim = 512
        self.p1 = 4. #4.
        self.p2 = 8.
        #self.p2 = 6.
        self.p3 = 2.
        self.loss = loss
        self.adj1_d1 = build_adj_full_d(8, int(self.p1), 1) 
        self.adj1_d2 = build_adj_full_d(8, int(self.p1), 2)
        self.adj1_d3 = build_adj_full_d(8, int(self.p1), 3)
        self.adj2_d1 = build_adj_full_d(8, int(self.p2), 1)
        self.adj2_d2 = build_adj_full_d(8, int(self.p2), 2)
        self.adj2_d3 = build_adj_full_d(8, int(self.p2), 3)
        self.adj3_d1 = build_adj_full_d(8, int(self.p3), 1)
        self.adj3_d2 = build_adj_full_d(8, int(self.p3), 2)
        self.adj3_d3 = build_adj_full_d(8, int(self.p3), 3)
        self.adj1_d1.requires_gradient = False
        self.adj2_d1.requires_gradient = False
        self.adj3_d1.requires_gradient = False
        self.adj1_d2.requires_gradient = False
        self.adj2_d2.requires_gradient = False
        self.adj3_d2.requires_gradient = False
        self.adj1_d3.requires_gradient = False
        self.adj2_d3.requires_gradient = False
        self.adj3_d3.requires_gradient = False


        #resnet50 = torchvision.models.resnet50(pretrained=True)
        #self.base = nn.Sequential(*list(resnet50.children())[:-2])

        #=============stride = 1 =======================
        #self.base = ResNet(last_stride=1,
        #                       block=Bottleneck,
        #                       layers=[3, 4, 6, 3])
        self.base = ResNetNonLocal(last_stride=1,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        self.base.load_param('/home/ubuntu/.torch/models/resnet50-19c8e357.pth')
        

        self.layers1 = nn.ModuleList([
            BatchedGraphSAGEDynamicRangeMean1(input_shape, self.hidden_dim),
            BatchedGraphSAGEDynamicRangeMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEDynamicMean1(input_shape, self.hidden_dim),
            #BatchedGraphSAGEDynamicMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEMean1(input_shape, self.hidden_dim),
            #BatchedGraphSAGEMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEMean1Temporal(input_shape, self.hidden_dim),
            #BatchedGraphSAGEMean1Temporal(4*self.hidden_dim, self.hidden_dim),
            #BatchedGAT_cat1(input_shape, self.hidden_dim),
            #BatchedGAT_cat1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGAT_cat1Temporal(input_shape, self.hidden_dim),
            #BatchedGAT_cat1Temporal(4*self.hidden_dim, self.hidden_dim),
        ])

        self.layers2 = nn.ModuleList([
            BatchedGraphSAGEDynamicRangeMean1(input_shape, self.hidden_dim),
            BatchedGraphSAGEDynamicRangeMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEDynamicMean1(input_shape, self.hidden_dim),
            #BatchedGraphSAGEDynamicMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEMean1(input_shape, self.hidden_dim),
            #BatchedGraphSAGEMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEMean1Temporal(input_shape, self.hidden_dim),
            #BatchedGraphSAGEMean1Temporal(4*self.hidden_dim, self.hidden_dim),
            #BatchedGAT_cat1(input_shape, self.hidden_dim),
            #BatchedGAT_cat1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGAT_cat1Temporal(input_shape, self.hidden_dim),
            #BatchedGAT_cat1Temporal(4*self.hidden_dim, self.hidden_dim),
        ])
 
        self.layers3 = nn.ModuleList([
            BatchedGraphSAGEDynamicRangeMean1(input_shape, self.hidden_dim),
            BatchedGraphSAGEDynamicRangeMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEDynamicMean1(input_shape, self.hidden_dim),
            #BatchedGraphSAGEDynamicMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEMean1(input_shape, self.hidden_dim),
            #BatchedGraphSAGEMean1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGraphSAGEMean1Temporal(input_shape, self.hidden_dim),
            #BatchedGraphSAGEMean1Temporal(4*self.hidden_dim, self.hidden_dim),
            #BatchedGAT_cat1(input_shape, self.hidden_dim),
            #BatchedGAT_cat1(2*self.hidden_dim, self.hidden_dim),
            #BatchedGAT_cat1Temporal(input_shape, self.hidden_dim),
            #BatchedGAT_cat1Temporal(4*self.hidden_dim, self.hidden_dim),
        ])
        #=======global================
        '''
        self.W = nn.Linear(self.input_shape, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        '''
        self.bn = nn.BatchNorm1d(self.input_shape)
        self.bnneck = nn.BatchNorm1d(self.input_shape + 3*2*self.hidden_dim)
        #self.bnneck = nn.BatchNorm1d(self.input_shape + 3*4*self.hidden_dim)

        self.classifier = nn.Linear(2*self.hidden_dim, n_classes)
        #self.classifier = nn.Linear(4*self.hidden_dim, n_classes)

        self.max_pool_fn = lambda x: x.max(dim=0)[0]

        #========bpm classifiers=================== 
        self.fc_list = nn.ModuleList()
        for i in range(4):
            if i == 0:
                classifier = nn.Linear(self.input_shape, n_classes)
            else:
                classifier = nn.Linear(2*self.hidden_dim, n_classes)
                #classifier = nn.Linear(4*self.hidden_dim, n_classes)
            nn.init.normal_(classifier.weight, std=0.001)
            nn.init.constant_(classifier.bias, 0)
            self.fc_list.append(classifier)

    #def forward(self, x, adj1, adj2, adj3):
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        
        #========global=============
        
        x0 = F.avg_pool2d(x, x.size()[2:])
        x0 = x0.view(b,t,-1)
        x0 = x0.permute(0,2,1)
        x0 = F.avg_pool1d(x0,t)
        x0 = x0.view(b, self.input_shape)
        #f0 = self.W(x0)
        f0 = F.normalize(x0, dim=1, p=2)
        f0 = F.relu(f0)
        f0 = self.bn(f0)        
        

        #========part p1============
        x1 = F.avg_pool2d(x, (int(x.size(-2)/self.p1), x.size(-1)))
        x1 = x1.permute(0,2,1,3)
        x1 = x1.contiguous().view(b, t, int(self.p1), -1)
        x1 = x1.view(b, t*int(self.p1), -1)
 
        for layer in self.layers1:
            #x1 = layer(x1, self.adj1_d1)
            x1 = layer(x1, self.adj1_d1, int(self.p1), 3)
            #x1 = layer(x1, self.adj1_d1, self.adj1_d2, self.adj1_d3)
        
        #======pooling p1=============
        f1 = x1.permute(0, 2, 1)
        f1 = F.avg_pool1d(f1, int(t*self.p1))     #comment for graph pool
        f1 = f1.view(b, 2*self.hidden_dim)
        #f1 = f1.view(b, 4*self.hidden_dim)

        #========part p2============
        x2 = F.avg_pool2d(x, (int(x.size(-2)/self.p2), x.size(-1)))
        x2 = x2.permute(0,2,1,3)
        x2 = x2.contiguous().view(b, t, int(self.p2), -1)
        x2 = x2.view(b, t*int(self.p2), -1)

        for layer in self.layers2:
            #x2 = layer(x2, self.adj2_d1)
            x2 = layer(x2, self.adj2_d1, int(self.p2), 3)
            #x2 = layer(x2, self.adj2_d1, self.adj2_d2, self.adj2_d3)

        #======pooling p2=============
        f2 = x2.permute(0, 2, 1)
        f2 = F.avg_pool1d(f2, int(t*self.p2))     #comment for graph pool
        f2 = f2.view(b, 2*self.hidden_dim)
        #f2 = f2.view(b, 4*self.hidden_dim)

        #========part p3============
        
        x3 = F.avg_pool2d(x, (int(x.size(-2)/self.p3), x.size(-1)))
        x3 = x3.permute(0,2,1,3)
        x3 = x3.contiguous().view(b, t, int(self.p3), -1)
        x3 = x3.view(b, t*int(self.p3), -1)

        for layer in self.layers3:
            #x3 = layer(x3, self.adj3_d1)
            x3 = layer(x3, self.adj3_d1, int(self.p3), 3)
            #x3 = layer(x3, self.adj3_d1, self.adj3_d2, self.adj3_d3)

        #======pooling p3=============
        f3 = x3.permute(0, 2, 1)
        f3 = F.avg_pool1d(f3, int(t*self.p3))     #comment for graph pool
        f3 = f3.view(b, 2*self.hidden_dim)
        #f3 = f3.view(b, 4*self.hidden_dim)
        

        #f = torch.cat((f0, f1, f2), 1)
        #f = torch.cat((f0, f3, f1, f2), 1)
        #f = (f0 + f1 + f2)/3
        #f = (f1 + f2)/2

        #f = torch.stack((f1,f2), dim=0)
        #f = torch.stack((f0, f1,f2), dim=0)
        #print(f.shape)
        #f = self.max_pool_fn(f)
        #print(f.shape)
    
        #=====bnneck==================
        #f = torch.cat((f0, f1, f2), 1)
        f = torch.cat((f0, f3, f1, f2), 1)
        #f = torch.cat((f0, f1), 1)
        f_bn = self.bnneck(f)
        
        #========bpm for different graph====================
        #local_feat = [f0, f3, f1, f2]
        #local_feat = [f_bn[:,0:self.input_shape], f_bn[:,self.input_shape:self.input_shape+2*self.hidden_dim], f_bn[:, self.input_shape+2*self.hidden_dim:]]
        local_feat = [f_bn[:,0:self.input_shape], f_bn[:,self.input_shape:self.input_shape+2*self.hidden_dim], f_bn[:, self.input_shape+2*self.hidden_dim:self.input_shape+4*self.hidden_dim], f_bn[:, self.input_shape+4*self.hidden_dim:]]
        #local_feat = [f_bn[:,0:self.input_shape], f_bn[:,self.input_shape:self.input_shape+4*self.hidden_dim], f_bn[:, self.input_shape+4*self.hidden_dim:self.input_shape+8*self.hidden_dim], f_bn[:, self.input_shape+8*self.hidden_dim:]]
        logits_list = []
        for i in range(4):
            logits_list.append(self.fc_list[i](local_feat[i]))

        if not self.training:
            #return f
            return f_bn
        #y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return logits_list, f #local_feat #f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

