import torch.nn as nn
# from torchlibrosa.stft import Spectrogram, LogmelFilterBank
# from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
# from pytorch_utils import do_mixup, interpolate, pad_framewise_output
__all__ = ['TSVAD1']

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


def conv_block(in_channels,out_channels,pooling_size):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=(1,3-pooling_size),padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(pooling_size)
    )

class TSVAD_LSTM(nn.Module):
    def __init__(self,num_classes):
        super(TSVAD_LSTM,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,128,pooling_size=2),
            conv_block(128,128,pooling_size=2),
            conv_block(128,128,pooling_size=1),
            conv_block(128,128,pooling_size=1)
        )
        self.lstm = nn.LSTM(431)
        self.fc = nn.Linear(128*8, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(128*8*2,2)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(128*8,2)
        )
        self.logsoftmax_fn = nn.LogSoftmax(dim=1)

    def forward(self,x,step=1):
        (num_samples,seq_len,mel_bin) = x.shape

        x = x.unsqueeze(1)
        x1 = x[:,:,:(seq_len//2)]
        x2 = x[:,:,(seq_len//2):]
        if step==0:
            x = torch.cat((x,torch.zeros_like(x[:,:,:1,:])),2)
            self.encoder[0].eval()
            self.encoder[1].eval()
            with torch.no_grad():
                x = self.encoder[0](x)
                x = self.encoder[1](x)

            x = self.encoder[2](x)
            x = self.encoder[3](x)
            x = x.repeat_interleave(4,dim=2)[:,:,:seq_len,:]
            x = x.permute(0,2,1,3).reshape(num_samples,-1,128*8)
            return x
        elif step==1:
            x1 = torch.cat((x1,torch.zeros_like(x1[:,:,:1,:])),2)
            x1 = self.encoder(x1)
            x1 = x1.repeat_interleave(4,dim=2)[:,:,:(seq_len//2),:]
            x1 = x1.permute(0,2,1,3).reshape(-1,128*8)
            pre = self.fc(x1)
        elif step==2:
            mask = torch.where(x2.sum(-1,keepdim=True)!=0,1,0)
            x1 = torch.cat((x1,torch.zeros_like(x1[:,:,:1,:])),2)
            x1 = self.encoder(x1)
            x1 = x1.repeat_interleave(4,dim=2)[:,:,:seq_len//2,:]
            x1 = x1.permute(0,2,1,3).reshape(num_samples,-1,128*8)

            x2 = self.forward_mask(x2,mask)
            x2 = x2.permute(0,2,1,3).reshape(num_samples,-1,128*8)

            vec = (x2*mask[:,0]).sum(1,keepdim=True)/mask[:,0].sum(1,keepdim=True)
            vec = vec.repeat(1,seq_len//2,1)
            cat_x = torch.cat((x1,vec),2).reshape(-1,128*8*2).contiguous()
            pre = self.decoder(cat_x)
        else:
            x1 = torch.cat((x1,torch.zeros_like(x1[:,:,:1,:])),2)
            x1 = self.encoder(x1)
            x1 = x1.repeat_interleave(4,dim=2)[:,:,:seq_len//2,:]
            x1 = x1.permute(0,2,1,3).reshape(-1,128*8)
            pre = self.decoder2(x1)
        return pre

    def forward_mask(self,x,mask):
        for i in range(4):
            x = self.encoder[i][0](x)*mask
            x = self.encoder[i][1](x)*mask
            x = self.encoder[i][2](x)
            if i<2:
                x = torch.functional.F.max_pool2d(input=x,kernel_size=(1,2))
        
        return x

    def forward_encoder_test(self,x):
        self.encoder.eval()
        with torch.no_grad():
            (num_samples,seq_len,mel_bins) = x.shape

            if seq_len<=4:
                x = torch.tile(x,[1,4//seq_len+1,1])[:,:4]
            elif seq_len%4==1:
                x = torch.cat((x,torch.zeros_like(x[:,:3,:])),1)
            elif seq_len%4==2:
                x = torch.cat((x,torch.zeros_like(x[:,:2,:])),1)
            elif seq_len%4==3:
                x = torch.cat((x,torch.zeros_like(x[:,:1,:])),1)
            x = x.unsqueeze(1)
            x = self.encoder(x)
            x = x.repeat_interleave(4,dim=2)[:,:,:seq_len,:]
            return x.permute(0,2,1,3).reshape(num_samples,seq_len,-1)
    
    def forward_decoder_test(self,vec,x):
        self.decoder.eval()
        with torch.no_grad():
            cat_x = torch.cat((x,vec.repeat(x.shape[0],1)), 1)
            pre = self.decoder(cat_x)
            logsoftmax = self.logsoftmax_fn(pre)

            return logsoftmax