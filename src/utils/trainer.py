import torch
import time
import torch.nn as nn
import numpy as np
from utils.util import warp_tqdm, get_metric, AverageMeter

# setting
print_freq = 100
meta_val_way = 10
meta_val_shot = 5
meta_val_metric = 'cosine'  # ('euclidean', 'cosine', 'l1', l2')
meta_val_iter = 500
meta_val_query = 15
alpha = - 1.0
label_smoothing = 0.
class Trainer:
    def __init__(self, device,num_class,train_loader,val_loader, conf):
        self.train_loader,self.val_loader = train_loader,val_loader
        self.device = device
        self.num_classes = num_class # 
        self.alpha = -1.0
        self.label_smoothing = 0.1
        self.meta_val_metric = 'cosine'
        self.loss_record={
            "20sepclass_loss":[],
            "2class_loss":[],
            "sep_loss":[]
        }
    def cross_entropy(self, logits, targets,mask, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        log_pos = torch.gather(logsoftmax,1,targets*mask)
        return - (log_pos * mask).sum()/(mask.sum()+10e-10)
   
    def cross_entropy2(self, logits, targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        log_pos = torch.gather(logsoftmax,1,targets)
        return -log_pos.mean()
    
    def do_epoch(self, epoch, scheduler, disable_tqdm, model,
                 alpha, optimizer,aug,logger_writer):
        batch_time = AverageMeter()
        losses_1 = AverageMeter()
        top1_1 = AverageMeter()

        losses = AverageMeter()
        top1 = AverageMeter()
       
        model.train()
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm) 

        for i, (input, target1, target2) in enumerate(tqdm_train_loader):
            # pdb.set_trace()
            ori_input, target1, target2 = input.to(self.device).squeeze(), target1.to(self.device, non_blocking=True).squeeze(), target2.to(self.device, non_blocking=True).squeeze()        
            target1 = target1.reshape(-1,1)
            target2 = target2.reshape(-1,1)
            mask = torch.where(target2>=0,1,0)
            if mask.sum()==0: continue # all overlap events no training
            
            if i!=0 and i%10==0:
                with torch.no_grad():
                   
                    show_image1 = ori_input[:,:431,:]-ori_input[:,:431,:].flatten(start_dim=-2).min(dim=1,keepdim=True).values.unsqueeze(-1)
                    show_image1 = ((show_image1/show_image1.flatten(start_dim=-2).max(dim=1,keepdim=True).values.unsqueeze(-1))*255).unsqueeze(1).repeat_interleave(3,dim=1)
                  
                    show_image2 = ori_input[:,431:,:]-ori_input[:,431:,:].flatten(start_dim=-2).min(dim=1,keepdim=True).values.unsqueeze(-1)
                    show_image2 = ((show_image2/show_image2.flatten(start_dim=-2).max(dim=1,keepdim=True).values.unsqueeze(-1))*255).unsqueeze(1).repeat_interleave(3,dim=1)
                    
                    show_image = torch.cat([show_image1,show_image2])
                    logger_writer.add_img("Input Image",show_image.permute([0,1,3,2]).type(torch.uint8),epoch*len(tqdm_train_loader)+i)
                    
            output_20cls = model(ori_input,step=1)
            output_vad = model(ori_input,step=3)
            
            loss1 = self.cross_entropy(output_20cls, target2,mask) # 20-classification
            loss2 = self.cross_entropy(output_vad,target1, mask) # 2-classification
            
            total_loss = loss1 + loss2
            if torch.isnan(total_loss):
                print(11)
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            prec1 = ((output_20cls.argmax(1) == target2.squeeze()).float()*mask.squeeze()).sum()/(mask.sum()+10e-10)
            prec2 = ((output_vad.argmax(1)==target1.squeeze()).float()*mask.squeeze()).sum()/(mask.sum()+10e-10)
            
            top1.update(prec1.item(), mask.sum().item())
            top1_1.update(prec2.item(),mask.sum().item())
            
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f} Acc_vad {:.2f}'.format(top1.avg, top1_1.avg))
            # Measure accurac y and record loss
            losses.update(total_loss.item(), mask.sum().item())
            losses_1.update(loss1.item(),mask.sum().item())
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss_al {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'Prec1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec_vad {top1_1.val:.3f} ({top1_1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, loss1=losses_1, top1=top1, top1_1=top1_1))

    def meta_val(self, model, disable_tqdm):
        top1 = AverageMeter()
        top1_1 = AverageMeter()
        model.eval() 

        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target1,target2) in enumerate(tqdm_test_loader): #
                inputs, target1, target2 = inputs.to(self.device), target1.to(self.device, non_blocking=True), target2.to(self.device, non_blocking=True)
                target1 = target1.reshape(-1,1)
                target2 = target2.reshape(-1,1)
                mask = torch.where(target2>=0,1,0)
                
                output_vad = model(inputs,step=3)
                output_20cls = model(inputs)
                acc_vad = ((output_vad.argmax(1)==target1.squeeze()).float()*mask.squeeze()).sum()/mask.sum() # 2 classification
                acc_20cls = ((output_20cls.argmax(1)==target2.squeeze()).float()*mask.squeeze()).sum() / mask.sum() #20 classification
                
                top1.update(acc_vad.item(),mask.sum().item())
                top1_1.update(acc_20cls.item(),mask.sum().item())
                if not disable_tqdm:
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg*100))
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1_1.avg*100))
        return top1.avg, top1_1.avg # 2-classification，20-classification
        