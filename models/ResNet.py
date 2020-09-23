from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from .convlstm import ConvLSTM
import math
__all__ = ['ResNet50TP', 'ResNet50TPICA', 'ResNet50TA', 'ResNet50RNN', 'ResNet50CONVRNN', 'ResNet50GRU', 'ResNet50TPNEW', 'ResNet50TPPART']
from .resnet import ResNet, BasicBlock, Bottleneck, ResNetNonLocal

class ResNet50TP(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        #self.base = nn.Sequential(*list(resnet50.children())[:-2])
        #===== res50 with stride = 1 ==================
        #self.base = ResNet(last_stride=1,
        #                       block=Bottleneck,
        #                       layers=[3, 4, 6, 3])
        self.base = ResNetNonLocal(last_stride=1,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        self.base.load_param('/home/yy1/.torch/models/resnet50-19c8e357.pth')
        #print(self.base.state_dict()['layer1.2.conv1.weight'])

        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50TPICA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TPICA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.hidden_dim = self.feat_dim/4
        self.out_dim = self.feat_dim/2

        #=======================
        self.batchnorm1 = nn.BatchNorm2d(self.feat_dim)
        self.lrelu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(self.feat_dim, self.hidden_dim, 1)
        self.batchnorm2 = nn.BatchNorm2d(self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.out_dim)
        #=======================
        self.classifier = nn.Linear(self.out_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = self.batchnorm1(x)
        x = self.lrelu(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.feat_dim)
        f = f.unsqueeze(2)
        f = f.unsqueeze(3)
        f = self.conv1(f)
        f = self.batchnorm2(f)
        f = f.view(b, self.hidden_dim)
        f = self.fc(f)
        
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50TPPART(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TPPART, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.p = 4.

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        #print(x.shape)
   
        #-----------------
        ''' 
        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(b,t,-1)
        x1=x1.permute(0,2,1)
        f1 = F.avg_pool1d(x1,t)
        f1 = f1.view(b, self.feat_dim)
        '''
        #-----------------

        #x = F.avg_pool2d(x, (int(math.ceil(x.size(-2)/self.p)), x.size(-1)), ceil_mode=True) # 128, 2048, 4, 1
        x = F.avg_pool2d(x, (2, x.size(-1)))

        #======================
        x = x.permute(0,2,1,3)
        x = x.contiguous().view(b, t, int(self.p), -1)
        x = x.view(b, t*int(self.p), -1)
        x = x.permute(0, 2, 1)
        f = F.avg_pool1d(x, t*int(self.p))
        f = f.view(b, self.feat_dim)
        #======================
        #print(x.shape)
        #x = x.view(b, t, x.size(1), x.size(2), -1)
        #x = x.permute(0, 2, 1)

        #x = x.view(b,int(t*self.p),-1)
        #print(x.shape)
        #f = x.permute(0, 2, 1)
        #f = F.avg_pool1d(f, int(t*self.p))
        #f = f.view(b, self.feat_dim)
        #print(f-f1)

        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant(m.weight, 1.0)
            nn.init.constant(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal(m.weight, std=0.001)
        if m.bias:
            nn.init.constant(m.bias, 0.0)

class ResNet50TPNEW(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TPNEW, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)

        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad = False

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = x.view(b,t,-1)
        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        f = att_x.view(b,self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50RNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50RNN, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=False)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x = x.permute(1, 0, 2)
        output, (h_n, c_n) = self.lstm(x)
        #print(output.shape)
        #output = output.permute(0, 2, 1)
        output = output.permute(1, 2, 0)
        #print(output.shape)
        f = F.avg_pool1d(output, t)
        #print(f.shape)
        f = f.view(b, self.hidden_dim)
        #print(f.shape)
        '''
        torch.Size([32, 4, 512])
        torch.Size([32, 512, 4])
        torch.Size([32, 512, 1])
        torch.Size([32, 512])
        '''
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50GRU(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50GRU, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = nn.GRU(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=False)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x = x.permute(1, 0, 2)
        output, h_n = self.lstm(x)
        #print(output.shape)
        #output = output.permute(0, 2, 1)
        output = output.permute(1, 2, 0)
        #print(output.shape)
        f = F.avg_pool1d(output, t)
        #print(f.shape)
        f = f.view(b, self.hidden_dim)
        #print(f.shape)
        '''
        torch.Size([32, 4, 512])
        torch.Size([32, 512, 4])
        torch.Size([32, 512, 1])
        torch.Size([32, 512])
        '''
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50CONVRNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50CONVRNN, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.input_size = 7
        self.feat_dim = 2048
        self.kernel_size = 3
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = ConvLSTM(input_size=(self.input_size, int(self.input_size/2 + 1)), input_dim=self.feat_dim, hidden_dim=self.hidden_dim, kernel_size=(self.kernel_size, self.kernel_size), num_layers=1, batch_first=False)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        #print(x.shape)
        #x = F.avg_pool2d(x, x.size()[2:])
        #x = x.view(b,t,-1)
        #x = x.permute(1, 0, 2)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3))
        x = x.permute(1, 0, 2, 3, 4)
         
        output, last_state = self.lstm(x)
        #print(output.shape)
        output = output[0]
        #print(output.shape)
        #output = output.permute(0, 2, 1)
        #output = output.permute(1, 2, 0)
        output = output.view(t*b, output.size(2), output.size(3), output.size(4))
        output = F.avg_pool2d(output, output.size()[2:])
        output = output.view(t, b, output.size(1))
        output = output.permute(1, 2, 0)
        #print(output.shape)
        f = F.avg_pool1d(output, t)
        #print(f.shape)
        f = f.view(b, self.hidden_dim)
        #print(f.shape)
        '''
        torch.Size([128, 2048, 7, 4])
        torch.Size([4, 32, 512, 7, 4])
        torch.Size([32, 512, 4])
        torch.Size([32, 512, 1])
        torch.Size([32, 512])
        '''
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

