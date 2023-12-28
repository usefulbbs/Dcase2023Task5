import pandas as pd
import random
import h5py
import pandas as pd
from glob import glob
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datasets.batch_sampler import EpisodicBatchSampler
from utils.trainer import Trainer
from utils.util import save_checkpoint
from models import __dict__
from datasets.Feature_extract import feature_transform
from datasets.Datagenerator import Datagen
from utils.eval import Evaluator
import time
import librosa
import json
import os
import os.path as osp
from src.utils.tensorbord_ import Summary

def get_model(arch,num_classes,ema=False):
    if arch == 'resnet10' or arch == 'resnet18':
        model = __dict__[arch](num_classes=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
            return model
        return model
    else:
        model = __dict__[arch](num_classes=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
            return model
        return model

def train_protonet(model,train_loader,valid_loader,conf,logger_writer):
    arch = 'Protonet'
    alpha = 0.0  
    disable_tqdm = True 
    ckpt_path = conf.eval.ckpt_path
    is_pretrain = False
    resume = False
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    optim = torch.optim.Adam(model.parameters(), lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    if is_pretrain: 
        pretrain = os.path.join(conf.path.work_path, 'check_point/checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))
            
    if resume:
        resume_path = ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1
    #cudnn.benchmark = True
    model.to(device,non_blocking=True) # cuda
    trainer = Trainer(device=device,num_class=conf.train.num_classes, train_loader=train_loader,val_loader=valid_loader,conf=conf)
    time_start=time.time()
    for epoch in range(num_epochs):
        trainer.do_epoch(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,alpha=alpha,optimizer=optim,aug=conf.train.aug,logger_writer=logger_writer)
        # Evaluation on validation set
        prec_vad, prec_20cls = trainer.meta_val(model=model, disable_tqdm=disable_tqdm)
        print('Meta vad_Val {}: {}'.format(epoch, prec_vad),'\t\t', 'Meta 20cls_Val {}: {}'.format(epoch, prec_20cls)) # 2-classification
        print() # 20-classification
        is_best = prec_vad > best_prec1
        best_prec1 = max(prec_vad, best_prec1)
        if not disable_tqdm:
            print('Best Acc {:.2f}'.format(best_prec1 * 100.))
        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': arch,
                               'state_dict': model.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optim.state_dict()},
                        is_best=is_best,
                        folder=ckpt_path)
        if lr_scheduler is not None:
            lr_scheduler.step()
    time_end=time.time()
    print('totally cost',time_end-time_start)
    print('model_paramiter...............')
    
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
        
def get_name2wav(wav_dir):
    hash_name2wav={}
    for subdir in os.listdir(wav_dir):
        sub_wav_dir = os.path.join(wav_dir,subdir)
        for name in os.listdir(sub_wav_dir):
            if ".wav" not in name:
                continue
            wav_path = os.path.join(sub_wav_dir,name)
            hash_name2wav[name] = wav_path
    return hash_name2wav

def get_name2wav2(wav_dir):
    hash_name2wav={}
    for wav_path in glob(osp.join(wav_dir,"*","*","*.wav")):
        name = wav_path.split("/")[-1]
        hash_name2wav[name] = wav_path
    return hash_name2wav

def setup_seed(seed):
    torch.manual_seed(seed=seed)
    os.environ['PTHONHASHSEED']=str(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark=False
    cudnn.deterministic = True
    cudnn.enabled = True 

@hydra.main(config_name="config")
def main(conf : DictConfig):
    seed = 2021
    if seed is not None:
        setup_seed(seed)
    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)
    if not os.path.isdir(conf.path.feat_train):
        os.makedirs(conf.path.feat_train)
    if not os.path.isdir(conf.path.feat_eval):
        os.makedirs(conf.path.feat_eval)
    if not os.path.isdir(conf.path.feat_test):
        os.makedirs(conf.path.feat_test)
    if not os.path.isdir(osp.join(conf.path.work_path,'src','train_data')):
        os.makedirs(osp.join(conf.path.work_path,'src','train_data'))
        
    if conf.set.features:
        print(" --Feature Extraction Stage--")
        Num_extract_train,data_shape = feature_transform(conf=conf,mode="train") # train data
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(Num_extract_train))
        
        Num_extract_eval = feature_transform(conf=conf,mode='eval')
        print("Total number of samples used for evaluation: {}".format(Num_extract_eval)) # validate data
        print(" --Feature Extraction Complete--")

        # Num_extract_test = feature_transform(conf=conf,mode='test')
        # print("Total number of samples used for evaluation: {}".format(Num_extract_test)) # test data
        # print(" --Feature Extraction Complete--")
        
    if conf.set.train: # train
        print("==========================================> start training <===============================================")
        meta_learning = False # wether use meta learing ways to train
        CURRENT_TIMES = time.strftime("%Y-%m-%d %H-%M",time.localtime())
        logger_writer = Summary(path=osp.join(conf.path.tensorboard_path,CURRENT_TIMES))
        # import pdb 
        # pdb.set_trace()
        if meta_learning:
            gen_train = Datagen(conf) 
            X_train,Y_train,X_val,Y_val = gen_train.generate_train() # 
            X_tr = torch.tensor(X_train) 
            Y_tr = torch.LongTensor(Y_train)
            X_val = torch.tensor(X_val)
            Y_val = torch.LongTensor(Y_val)

            samples_per_cls =  conf.train.n_shot * 2 

            batch_size_tr = samples_per_cls * conf.train.k_way # the batch size of training 
            batch_size_vd = batch_size_tr # 

            num_batches_tr = len(Y_train)//batch_size_tr # num of batch
            num_batches_vd = len(Y_val)//batch_size_vd

            samplr_train = EpisodicBatchSampler(Y_train,num_batches_tr,conf.train.k_way,samples_per_cls) # batch_size_tr = samples_per_cls * conf.train.k_way
            samplr_valid = EpisodicBatchSampler(Y_val,num_batches_vd,conf.train.k_way,samples_per_cls)

            train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr)
            valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=samplr_train,shuffle=False)
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,shuffle=False)
        else: 
            try:
                train_datasets = torch.load(conf.train.train_data)
                X_tr=train_datasets['X_tr']
                Y_tr=train_datasets['Y_tr']
                Y2_tr=train_datasets['Y2_tr']
                X_val=train_datasets['X_val']
                Y_val=train_datasets['Y_val']
                Y2_val=train_datasets['Y2_val']
            except:
                gen_train = Datagen(conf) 
                X_train,Y_train,Y2_train,X_val,Y_val,Y2_val = gen_train.generate_train() 
                X_tr = torch.tensor(X_train) 
                Y_tr = torch.LongTensor(Y_train)
                Y2_tr = torch.LongTensor(Y2_train)
                
                X_val = torch.tensor(X_val)
                Y_val = torch.LongTensor(Y_val)
                Y2_val = torch.LongTensor(Y2_val)
                state = {'X_tr':X_tr, 'Y_tr':Y_tr, 'Y2_tr':Y2_tr, 
                         'X_val':X_val, 'Y_val':Y_val,'Y2_val':Y2_val}
                torch.save(state,conf.train.train_data)
        samples_per_cls =  conf.train.n_shot 
        batch_size_tr =  conf.train.n_shot* conf.train.k_way #64 # the batch size of training 
        
        num_batches_tr = len(Y_tr)//batch_size_tr
        ## meta training
        samplr_train = EpisodicBatchSampler(Y_tr,Y2_tr,num_batches_tr,conf.train.k_way,samples_per_cls) 

        train_dataset = torch.utils.data.TensorDataset(X_tr,Y2_tr,Y_tr) # X,Y2,Y, Y2∈{0,1}, Y∈{0,1,2....,19}
        valid_dataset = torch.utils.data.TensorDataset(X_val,Y2_val,Y_val)
        
        if conf.train.meta_training==True:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=samplr_train, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_tr
                                                       , num_workers=10, pin_memory=True, shuffle=True)
            
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=64,
                                                   batch_sampler=None, num_workers=10, pin_memory=True, shuffle=True)
    
        logger_writer = Summary(path=osp.join(conf.path.tensorboard_path, CURRENT_TIMES))
        model = get_model('TSVAD1',conf.train.num_classes)
        train_protonet(model,train_loader, valid_loader,conf,logger_writer)
                    
    if conf.set.eval: # eval
        k_q = 128
        device = 'cuda'
        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])

        name_arr_ori = np.array([])
        onset_arr_ori = np.array([])
        offset_arr_ori = np.array([])

        all_feat_files = sorted([file for file in glob(os.path.join(conf.path.feat_eval,'*.h5'))])
        evaluator = Evaluator(device=device)
        model = get_model('TSVAD1',conf.train.num_classes).cuda()
      
        ckpt_path = conf.eval.ckpt_path
        hash_name2wav = get_name2wav(conf.path.eval_dir)
        hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel) # 0.05*22050//256 == 86

        TOTAL_LENGTH = len(all_feat_files)
        
        for i in range(TOTAL_LENGTH):
            feat_file = all_feat_files[i]
            feat_name = feat_file.split('/')[-1]   
            file_name = feat_name.replace('.h5','')
           
            audio_name = feat_name.replace('h5','wav')
            print("Processing audio file : {}".format(audio_name))
           
            wav_path = hash_name2wav[audio_name]

            ori_path = os.path.join(conf.path.work_path,'src','output_csv','ori')
            if not os.path.exists(ori_path):
                os.makedirs(ori_path)
            tim_path = os.path.join(conf.path.work_path,'src','output_csv','tim')
            if not os.path.exists(tim_path):
                os.makedirs(tim_path)   
            if os.path.exists(os.path.join(conf.path.work_path,'waveFrame',file_name,"waveFrame.json")):
                reader = open(os.path.join(conf.path.work_path,'waveFrame',file_name,"waveFrame.json"),'r')
                waveData = json.load(fp=reader)
                nframe = waveData['nframe']
            else:
               if not os.path.exists(os.path.join(conf.path.work_path,'waveFrame',file_name)):
                    os.makedirs(os.path.join(conf.path.work_path,'waveFrame',file_name))
               writer = open(os.path.join(conf.path.work_path,'waveFrame',file_name,'waveFrame.json'),'w')
               y, fs = librosa.load(wav_path,sr= conf.features.sr)
               nframe = len(y)//conf.features.hop_mel
               json.dump({'nframe':nframe},fp=writer)
               writer.close()

            hdf_eval = h5py.File(feat_file,'r')
            start_index_query =  hdf_eval['start_index_query'][:][0]
            
            fileDict = {}
            fileDict['nframes']=nframe
            fileDict['start_index_query']=start_index_query
            result, num,_= evaluator.run_full_evaluation(test_file=audio_name[:-4],model=model,
                                                         model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf) # only update W
            predict = result[0]
            thre = max(result[2]-0.05,0.5)
            mean_pos_len = result[3]

            krn = np.array([1,-1])
            prob_thresh = np.where(predict>thre,1,0)

            prob_middle = flatten_res(prob_thresh,hop_seg,int(nframe-start_index_query))
            predict_middle = flatten_res(predict,hop_seg,int(nframe-start_index_query))

            changes_smooth = np.convolve(krn, prob_middle)
            onset_frames_smooth = np.where(changes_smooth==1)[0] # start point
            offset_frames_smooth = np.where(changes_smooth==-1)[0] # end point
            
            for i in range(onset_frames_smooth.shape[0]-1):
                start = offset_frames_smooth[i]
                end = onset_frames_smooth[i+1]
                if mean_pos_len > 2*87 and end-start<87 and predict_middle[start:end].mean()>0.5:
                    prob_middle[start:end]=1
                    
            prob_med_filt = medFilt(prob_middle,5)
            start_index_query = start_index_query*conf.features.hop_mel / conf.features.sr
            print('start_index_query',start_index_query)

            ###############################ORI############################
            changes_ori = np.convolve(krn,prob_middle) # 
            onset_frames_ori = np.where(changes_ori==1)[0]
            offset_frames_ori = np.where(changes_ori==-1)[0]

            onset_ori = (onset_frames_ori+1) *conf.features.hop_mel / conf.features.sr
            onset_ori = onset_ori + start_index_query
            offset_ori = (offset_frames_ori+1) *conf.features.hop_mel / conf.features.sr
            offset_ori = offset_ori + start_index_query

            name_ori = np.repeat(audio_name,len(onset_ori))
            name_arr_ori = np.append(name_arr_ori,name_ori)
            onset_arr_ori = np.append(onset_arr_ori,onset_ori)
            offset_arr_ori = np.append(offset_arr_ori,offset_ori)
            #############################################################

            changes = np.convolve(krn, prob_med_filt) # Median filter
            onset_frames = np.where(changes==1)[0]
            print("onset_frames", onset_frames.shape,'\n')
            offset_frames = np.where(changes==-1)[0]

            onset = (onset_frames+1) * conf.features.hop_mel / conf.features.sr
            onset = onset + start_index_query 
            offset = (offset_frames+1) * conf.features.hop_mel / conf.features.sr
            offset = offset + start_index_query

            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out_ori = pd.DataFrame({'Audiofilename':name_arr_ori,'Starttime':onset_arr_ori,'Endtime':offset_arr_ori})
        csv_path_ori = os.path.join(conf.path.work_path,'src','output_csv','ori','Eval_out_ori_1.csv')
        df_out_ori.to_csv(csv_path_ori,index=False)

        df_out_tim = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path_tim = os.path.join(conf.path.work_path,'src','output_csv','tim','Eval_out_tim_1.csv')
        df_out_tim.to_csv(csv_path_tim,index=False)

    if conf.set.test: # It only be used when test the final dataset of DCASE2021 task5

        device = 'cuda'

        # init_seed()
        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        all_feat_files = sorted([file for file in glob(os.path.join(conf.path.feat_test,'*.h5'))])
        evaluator = Evaluator(device=device)
        model = get_model('Protonet',19).cuda()
        student_model = get_model('Protonet',19).cuda()
        ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/check_point'
        hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel) # 0.05*22050//256 == 4
        k_q = 128
        iter_num = 0
        for feat_file in all_feat_files[:1]:
            print('file name ',feat_file)
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5','wav')
            print("Processing audio file : {}".format(audio_name))
            hdf_eval = h5py.File(feat_file,'r')
            strt_index_query =  hdf_eval['start_index_query'][:][0]
            result, num= evaluator.run_full_evaluation(test_file=audio_name[:-4],model=model,student_model=student_model,
                                                        model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf,k_q=k_q,iter_num=iter_num)
            predict = result[0]
            if predict.shape[0]>num:
                n_ = predict.shape[0]//k_q
                print('n_ ',n_)
                prob_final = predict[:(n_-1)*k_q]
                n_last = num - prob_final.shape[0]
                print('n_last ',n_last)
                prob_final = np.concatenate((prob_final,predict[-n_last:]))
                print('prob_final ',prob_final.shape)
            else:
                prob_final = predict
            
            assert num == prob_final.shape[0]
            krn = np.array([1, -1])
            prob_thresh = np.where(prob_final > 0.5, 1, 0) # 70572
            prob_pos_final = prob_final * prob_thresh
            changes = np.convolve(krn, prob_thresh) # 70573
            onset_frames = np.where(changes == 1)[0]
            print('onset_frames ',onset_frames.shape)
            offset_frames = np.where(changes == -1)[0]
            str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr 
            print('str_time_query ',str_time_query) # 322.5
            onset = (onset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
            onset = onset + str_time_query
            offset = (offset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
            offset = offset + str_time_query
            assert len(onset) == len(offset)
            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path = os.path.join(conf.path.work_path,'Eval_out_tim_test.csv')
        df_out.to_csv(csv_path,index=False)

def medFilt(detections, median_window):

    if median_window %2==0:
        median_window-=1
    x = detections
    k = median_window
    assert k%2 == 1, "Median filter length must be odd"
    assert x.ndim == 1, "Input must be one dimensional"
    k2 = (k - 1) // 2
    y = np.zeros((len(x),k),dtype=x.dtype)
    y[:,k2]=x
    for i in range(k2):
        j = k2 - i
        y[j:,i]=x[:-j]
        y[:j,i]=x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median(y,axis=1)

def medFilt0(detections, median_window,hop_seg):

    if median_window %2==0:
        median_window-=1
    x = detections
    k = median_window

    seg_num, seg_len = x.shape
   
    if k%2==0:
        k-=1
    k2 = (k-1)//2
    y = np.zeros((seg_num*hop_seg+seg_len-hop_seg,k), dtype=x.dtype)

    for j in range(k):
        for i in range(seg_num):
            start_y = i+j
            if i<j:
                y[i*hop_seg:(i+1)*hop_seg,j] = y[i*hop_seg:(i+1)*hop_seg,j-1]
            elif start_y >=seg_num:
                continue
            else:
                y[start_y*hop_seg:(start_y+1)*hop_seg,j] = x[i,j*hop_seg:(j+1)*hop_seg]
        y[seg_num*hop_seg:,j]=x[-1,-(seg_len-hop_seg):]
    return np.median(y,axis=1)

def flatten_res(detections,hop_seg,nframe):
    x = detections
    seg_num, _ = x.shape

    y = np.zeros(nframe,dtype=x.dtype)
    for i in range(seg_num-3):
        y[(i+2)*hop_seg:(i+3)*hop_seg] = x[i,2*hop_seg:3*hop_seg]
    y[:hop_seg] = x[0,:hop_seg]
    y[hop_seg:2*hop_seg] = x[0,hop_seg:2*hop_seg]
    y[(seg_num-1)*hop_seg:nframe] = x[seg_num-1,-(nframe-(seg_num-1)*hop_seg):]
    return y

if __name__ == '__main__':   
    main()


