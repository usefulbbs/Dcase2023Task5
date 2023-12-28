from utils.util import get_mi, get_cond_entropy, get_entropy, get_one_hot
from tqdm import tqdm
import torch
import time
import torch.nn.functional as F
import logging
import os
import math
import numpy as np
import torch.nn as nn 
import random

class TIM(object):
    def __init__(self,model, test_file, first):
        self.lr = 1e-3
        self.temp = 0.1 # different model may need different temp value
        self.loss_weights = [0.1, 0.1, 0.1] # [0.1, 0.1, 1]
        #self.loss_weights = [1, 1, 1] # [0.1, 0.1, 1]
        self.model = model 
        self.alpha = 1.0
        self.init_info_lists()
        
        self.m = 0.4
        self.s = 64
        self.cos_m =math.cos(self.m)
        self.sin_m =math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
    
    def init_info_lists(self):
        self.timestamps = []
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_probs = []
        self.losses = []
        
    def get_logits(self, samples, is_train=False, is_class=False,label=None, embedding=False):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        model = self.model
        bias = self.bias[:,0]
        b_s = 32
        if is_train:
            model.train()
            batch_size,win,ndim = samples.shape
            list_vec = []
            for i in np.arange(0, batch_size,b_s):
                sample = samples[i:i+b_s]
                list_vec.append(model(sample,step=0))
                outputs_samples = torch.cat(list_vec,0)
            if is_class:
                logits = model.fc(outputs_samples)
                return logits
        else:
            list_vec = []
            model.eval()
            with torch.no_grad():
                batch_size, win, ndim = samples.shape
                for i in np.arange(0,batch_size,b_s):
                    sample = samples[i:i+b_s].reshape(-1,win,ndim)
                    list_vec.append(model.forward_encoder_test(sample))
                outputs_samples = torch.cat(list_vec,0)
        
        if None == label:
            logits0 = outputs_samples.matmul(self.weights[:,0:1].transpose(1,2)) + bias
            logits1 = outputs_samples.matmul(model.fc.weight[0].view(1,-1,1)) + model.fc.bias[0].view(1,1,-1)
            logits = torch.cat((logits0,logits1),-1)
        else:
            cosine = F.normalize(outputs_samples,dim=2).matmul(F.normalize(self.weights.transpose(1,2),dim=1))
            sine = torch.sqrt((1.0-torch.pow(cosine,2)).clamp(0,1))
            phi = cosine * self.cos_m - sine*self.sin_m
            phi - torch.where(cosine > self.th, phi, cosine-self.mm)
            one_hot = label
            output = (one_hot * phi) + ((1.0 - one_hot)*cosine)
            logits = output*self.s
        if embedding:
            return logits ,outputs_samples
        else:
            return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]
        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def get_acc(self, preds,label,mask=None):
        if mask == None:
            acc = (preds.argmax(2)==label).int().sum()/label.numel()
        else:
            acc = (mask*(preds.argmax(1,keepdim=True)==label).int()).sum()/mask.sum()
        return acc

    def compute_FB_param(self, features_q):
        logits_q = self.get_logits(features_q) # logits: according to W, calculate results
        logits_q = logits_q.detach()
        q_probs = logits_q.softmax(2) # predict probability
        #probas = self.get_probas(features_q).detach()
        b = q_probs[:,:,0]>0.5
        # print(1.0*b.sum(1)/a.shape[1])
        pos = 1.0*b.sum(1)/q_probs.shape[1]
        neg = 1.0 -pos
        pos = pos.unsqueeze(1)
        neg = neg.unsqueeze(1)
        self.FB_param2 = torch.cat([pos,neg],1)
        self.FB_param = (q_probs).mean(dim=1)

    def init_weights(self, support, query, y_s,sub_train,y_t): 
        self.model.eval()
        t0 = time.time()
        n_tasks = support.size(0)
        with torch.no_grad():
            outputs_support = self.model.forward_encoder_test(support)
            outputs_support = F.normalize(outputs_support, dim=2)
        # print('n_tasks ',n_tasks)
        one_hot = get_one_hot(y_s) # get one-hot vector
        counts = one_hot.sum(1).view(n_tasks, -1, 1) # 
        
        weights = one_hot.transpose(1, 2).matmul(outputs_support) 
        self.weights = weights.sum(0,keepdim=True) / counts.sum(0,keepdim=True)

        self.weights = F.normalize(self.weights,dim=2)
        self.bias  = torch.tensor([0.1]).reshape(1,1,1).type_as(self.weights)

        self.model.fc.weight[0].data.copy_(self.weights[0,1])
        # self.model_student.fc.weight[0].data.copy_(self.weights[0,1])
        self.weights = self.weights[:,0:1,:]

        self.record_info(new_time=time.time()-t0,
                         support=support,
                         query=query,
                         y_s=y_s)
        self.model.train()

    def compute_lambda(self, support, query, y_s): 
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]

        updates :
            self.loss_weights[0] : Scalar
        """
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0) 
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q

    def record_info(self, new_time, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        with torch.no_grad():
            logits_s = self.get_logits(support)
            logits_s = logits_s.softmax(2).detach() 

            self.thre = ((logits_s[:,:,1]*y_s).sum(1)/y_s.sum(1)).min().item()
        
            logits_q = self.get_logits(query)
            logits_q = logits_q.softmax(2).detach()
            q_probs = logits_q 
               
            self.mutual_infos.append(get_mi(probs=q_probs.detach().cpu())) 
            self.entropy.append(get_entropy(probs=q_probs.detach().cpu())) # # H(Y_q)
            self.cond_entropy.append(get_cond_entropy(probs=q_probs.detach().cpu())) # # H(Y_q | X_q)
            self.test_probs.append(q_probs[:,:,1].cpu()) 
            self.y_s = y_s.cpu().numpy()
            torch.cuda.empty_cache()

    def get_logs(self):
        self.test_probs = self.test_probs[-1].cpu().numpy() # use the last as results
        self.cond_entropy = torch.cat(self.cond_entropy, dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy, dim=1).cpu().numpy()
        self.mutual_infos = torch.cat(self.mutual_infos, dim=1).cpu().numpy()
        self.W = self.weights
        return {'timestamps': self.timestamps, 'mutual_info': self.mutual_infos,
                'entropy': self.entropy, 'cond_entropy': self.cond_entropy, 'losses': self.losses,
                'test': self.test_probs,'W': self.W,'thre':self.thre,'y_s':self.y_s}

class TIM_GD(TIM):
    def __init__(self,model, test_file, first):
        super().__init__(model=model, test_file=test_file, first=first)

    def run_adaptation(self, support, query, y_s, min_len, sub_train, y_t, mask_t):
        t0 = time.time()
        self.min_len = min_len
        self.weights.requires_grad_() # W
        self.bias.requires_grad_() 
        optimizer = torch.optim.Adam([
            {'params':self.model.fc.parameters(),'lr':self.lr},
            {'params':self.model.encoder[2].parameters(),'lr':0.1*self.lr},
            {'params':self.model.encoder[3].parameters(),'lr':0.1*self.lr}, 
            {'params':self.weights},{'params':self.bias}], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.eval()
        l3 = 0.2
        self.iter=100
        for i in range(self.iter): # 
            start = time.time()
            self.model.train()
            choice_id = random.choice(range(4))
            select_sub_train = torch.cat((sub_train[choice_id:24*19:4],sub_train[24*19+choice_id*2::4*2]),0).contiguous().cuda()
            select_y_t = torch.cat((y_t[choice_id:24*19:4],y_t[24*19+choice_id*2::4*2]),0).contiguous().cuda()
            select_mask_t = torch.cat((mask_t[choice_id:24*19:4],mask_t[24*19+choice_id*2::4*2]),0).contiguous().cuda()

            select_y_t = select_y_t.reshape(-1,1)
            logits_t = self.get_logits(select_sub_train,is_train=True,is_class=True).reshape(-1,20)
            select_mask_t = select_mask_t.reshape(-1,1)
            
            ce_t = self.cross_entropy(logits_t,select_y_t,select_mask_t)
            # get_support_vec 
            logits_s = self.get_logits(support, is_train=True)
            ce = -(y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            # query
            logits_q = self.get_logits(query) #  
            q_probs = logits_q.softmax(2)
            self.select_query_data(q_probs[:,:,1],query,thre=self.thre)          
            #pseudo-label
            if len(self.torch_q_x)>0 and i>=85: #   
                mask = torch.where(self.torch_q_y==-1, 0, 1).cuda()
                y_qs_one_hot = get_one_hot(self.torch_q_y.cuda()*mask)
                logits_qs = self.get_logits(self.torch_q_x.cuda(),is_train=True,embedding=False)
                logits_qs = logits_qs.softmax(2) #.unsqueeze(-1)
                ce_qs = -(mask.unsqueeze(2)*y_qs_one_hot * torch.log(logits_qs+1e-12)).sum(2).mean(1).sum(0)
                
                self.loss_weights[1] = 0.1
            else:
                ce_qs = 0
                self.loss_weights[1]=0 
                
            self.loss_weights[2]=0.1  #if i <50 else 0.1
            loss = self.loss_weights[0] * ce + self.loss_weights[0]*ce_t + self.loss_weights[1]*ce_qs
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s_acc = self.get_acc(logits_s,y_s)
            t_acc = self.get_acc(logits_t.detach(),select_y_t,select_mask_t)  
            
            if i > 2:
                self.compute_FB_param(query)
                l3 += 0.1
            t1 = time.time()
            self.model.eval()
            self.record_info(new_time=t1-t0,
                            support=support,
                            query=query,
                            y_s=y_s)    
            batch_time = t1-start
            if i==0 or i%20==0 or i==self.iter-1: 
                print('iter: [{0}/{1}]\t'
                        'Time {batch_time:.3f} \t'
                        'Loss1 [{loss:.3f},{loss2:.3f}]\t'
                        'Prec [{s:.3f},{t:.3f}]\t'
                        'Thres: {thr:.3f}\t'
                        # 'Num_n_p: [{N_n},{N_q}] \t'
                        'Num_q_x: {N_qx}\t'.format(
                        i+1, self.iter, batch_time=batch_time,
                        loss=loss.item()*self.loss_weights[0], loss2 = -1,
                        s=s_acc, t=t_acc, thr=self.thre*100,N_qx=len(self.torch_q_x)))

    def cross_entropy(self,logits,targets,mask):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        log_pos = torch.gather(logsoftmax,1,(targets*mask).long())
        return -(log_pos*mask).sum()/mask.sum()


    def select_query_data(self,q_probs,query,thre):
        list_x_n = []
        list_y_n = []
        list_x_p = []
        list_y_p = []
        cnt_n = 0
        cnt_p = 0
        L = torch.randperm(q_probs.shape[0])
        for i in L:
            if self.min_len>2*87:
                sub_q_probs = self.meanFilt(q_probs[i],87)
            else:
                sub_q_probs = q_probs[i]

            p_index = (sub_q_probs>thre).long()
            n_index = (sub_q_probs<thre-0.2).long()

            p_index = self.medFilt(p_index,5)
            n_index = self.medFilt(n_index,5)
            np_index = (n_index+p_index-1)+p_index

            if n_index.sum()>0 and cnt_n<128:
                list_x_n.append(query[i])
                list_y_n.append(np_index)
                cnt_n +=1
            if p_index.sum()>0 and cnt_p<128:
                list_x_p.append(query[i])
                list_y_p.append(np_index)
                cnt_p +=1

        cnt = min(cnt_n,cnt_p)
        if cnt>0 and cnt==cnt_n:
            list_x = list_x_n[:cnt]
            list_y = list_y_n[:cnt]
        elif cnt>0 and cnt==cnt_p:
            list_x = list_x_p[:cnt]
            list_y = list_y_p[:cnt]
        elif cnt_n==0:
            list_x = list_x_p[:5]
            list_y = list_y_p[:5]
        else:
            list_x = list_x_n[:5]
            list_y = list_y_n[:5]

        if len(list_x)>0:
            self.torch_q_x = torch.stack(list_x)
            self.torch_q_y = torch.stack(list_y).long()
        else:
            self.torch_q_x = []
            self.torch_q_y = []
              
    def medFilt(self,detections, median_window):

        if median_window %2==0:
            median_window-=1

        x = detections
        k = median_window

        assert k%2 == 1, "Median filter length must be odd"
        assert x.ndim == 1, "Input must be one dimensional"
        k2 = (k - 1) // 2
        y = torch.zeros((len(x),k)).type_as(x)
        y[:,k2]=x
        for i in range(k2):
            j = k2 -1
            y[j:,i]=x[:-j]
            y[:j,i]=x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]

        return torch.median(y,axis=1)[0]

    def meanFilt(self,detections, median_window):

        if median_window %2==0:
            median_window-=1

        x = detections
        k = median_window

        assert k%2 == 1, "Median filter length must be odd"
        assert x.ndim == 1, "Input must be one dimensional"
        k2 = (k - 1) // 2
        y = torch.zeros((len(x),k)).type_as(x)
        y[:,k2]=x
        for i in range(k2):
            j = k2 -1
            y[j:,i]=x[:-j]
            y[:j,i]=x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]

        return torch.mean(y,axis=1)
