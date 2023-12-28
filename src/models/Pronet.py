import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
# from pytorch_utils import do_mixup, interpolate, pad_framewise_output
__all__ = ['Protonet_CLR','Protonet','Protonet3','Protonet_sep']

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if x.shape[2]<2:
            return x
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


def conv_block(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Protonet(nn.Module):
    def __init__(self,num_classes):
        super(Protonet,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,128),
            conv_block(128,128),
            conv_block(128,128),
            conv_block(128,128)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self,x,feature=False):
        (num_samples,seq_len,mel_bins) = x.shape
        # print('x ',x.shape)
        x = x.view(-1,1,seq_len,mel_bins)
        # print('x ',x.shape)
        x = self.encoder(x) # 100,128,1,8
        # print('x2 ', x.shape) 
        x = x.view(x.size(0),-1)
        pre = self.fc(x)
        if feature:
            return x,pre
        return pre

class Protonet3(nn.Module):
    def __init__(self):
        super(Protonet3,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,128),
            conv_block(128,128),
            conv_block(128,128)
        )
        self.fc = nn.Linear(1024, 19)
        self.mp = nn.MaxPool2d(2)

    def forward(self,x,feature=False):
        (num_samples,seq_len,mel_bins) = x.shape
        # print('x ',x.shape)
        x = x.view(-1,1,seq_len,mel_bins)
        # print('x ',x.shape)
        x = self.encoder(x) # 100,128,1,8
        #print('x2 ', x.shape) 
        x = self.mp(x)
        x = x.view(x.size(0),-1)
        pre = self.fc(x)
        if feature:
            return x,pre
        return pre

class Protonet_CLR(nn.Module):
    def __init__(self):
        super(Protonet_CLR,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,128),
            conv_block(128,256),
            conv_block(256,256),
            conv_block(256,128)
        )
        self.projection = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 512, bias=True))
                               
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape
        # print('x ',x.shape)
        x = x.view(-1,1,seq_len,mel_bins)
        # print('x ',x.shape)
        feature = self.encoder(x) # 100,128,1,8
        feature = feature.view(x.size(0),-1)
        # print('feature ',feature.shape)
        out = self.projection(feature)
        # print('x2 ', x.shape) 
        return feature, out

    
class Protonet_sep(nn.Module):
    def __init__(self,num_classes=1) -> None:
        super().__init__()
        convs = [
            #cnn1
            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv2d(1,64,kernel_size=(1,7), dilation=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #cnn2
            nn.ZeroPad2d((0,0,3,3)),
            nn.Conv2d(64,64,kernel_size=(7,1), dilation=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64,64,kernel_size=(5,5), dilation=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64,64,kernel_size=(5,5), dilation=(2,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64,64,kernel_size=(5,5), dilation=(4,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #cnn6
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64,64,kernel_size=(5,5), dilation=(8,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #cnn7
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64,64,kernel_size=(5,5), dilation=(16,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #cnn8
            nn.Conv2d(64,8,kernel_size=(1,1), dilation=(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU()]

        self.conv = nn.Sequential(*convs)

        self.lstm = nn.LSTM(
            8*128 +1024,
            512,
            batch_first=True,
            bidirectional=True)
        self.fc1 = nn.Linear(2*512, 512)
        self.fc2 = nn.Linear(512,128)

    def forward(self,x, speaker_embedding,step):
        n_samples, seq_len, mel_bin = x.shape

        x = x.unsqueeze(1)

        x = self.conv(x)

        x = x.transpose(1,2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        if step==0:
            repeat_num = x.shape[0]//speaker_embedding.shape[0]
            # speaker_embedding = speaker_embedding.unsqueeze(1)
            speaker_embedding = torch.repeat_interleave(speaker_embedding,repeat_num,dim=0)
            speaker_embedding = speaker_embedding.repeat(1,x.size(1),1)

            x = torch.cat((x,speaker_embedding), dim=2)
            x,_ = self.lstm(x)
            x = F.relu(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)
        if step==1: #fine_tune
            
            speaker_embedding = speaker_embedding.repeat(1,x.size(1),1)
            x = torch.cat((x,speaker_embedding), dim=2)
            x,_ = self.lstm(x)
            x = F.relu(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)
        return x

    def forward_test(self,x,speaker_embedding):
        with torch.no_grad():
            n_samples, seq_len, mel_bin = x.shape
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.transpose(1,2).contiguous()
            x = x.view(x.size(0), x.size(1), -1)

            speaker_embedding = speaker_embedding.unsqueeze(1)
            speaker_embedding = speaker_embedding.repeat(1,x.size(1),1)

            x = torch.cat((x,speaker_embedding), dim=2)
            x,_ = self.lstm(x)
            x = F.relu(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)
            return x