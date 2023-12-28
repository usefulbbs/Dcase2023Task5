import numpy as np
from utils.util import warp_tqdm, load_checkpoint
from utils.util import load_pickle, save_pickle
import os
import torch
import torch.nn.functional as F
from utils.tim import TIM, TIM_GD
from datasets.Datagenerator import Datagen_test,Datagen_train_select
from utils.util import warp_tqdm
import torch.nn as nn
import random

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.number_tasks = 10000
        self.n_ways = 2
        self.query_shots = 15
        self.method = 'tim_gd' #tim_gd
        self.model_tag = 'best'
        self.plt_metrics = ['accs']
        self.shots = [5]
        self.used_set = 'test'
        self.fresh_start = True

    def run_full_evaluation(self,test_file, model, model_path, hdf_eval, conf):
        """
        Run the evaluation over all the tasks in parallel
        inputs:
            test_file: The name of test audio file.
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            method : Which method to use for inference ("baseline", "tim-gd" or "tim-adm")
            shots : Number of support shots to try

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        print("=> Runnning full evaluation with method: {}".format(self.method))
        print("=> Load model from: {}".format(model_path))
        load_checkpoint(model=model, model_path=model_path, type=self.model_tag)
        # Get loaders

        loaders_dic, _ = self.get_loaders(hdf_eval=hdf_eval,conf=conf) 
        # Extract features (just load them if already in memory)
        extracted_features_dic = self.extract_features(model=model, model_path=model_path, model_tag=self.model_tag,
                                    used_set=test_file, fresh_start=self.fresh_start,loaders_dic=loaders_dic)
        results = []
        predict = None
        
        tasks,_ = self.generate_tasks(extracted_features_dic=extracted_features_dic, conf=conf,
                                model=model, loaders_dic=loaders_dic)  
        logs = self.run_task(task_dic=tasks, model=model,test_file=test_file,first=0)
        
        predict = logs['test']
        W = logs['W']
        thre = logs['thre']
        results.append(predict)
        results.append(tasks['MFL'])
        results.append(thre)
        results.append(tasks['mean_pos_len'])
        
        return results, self.number_tasks,logs

    def cross_entropy(self, logits, one_hot_targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        return - (one_hot_targets * logsoftmax).sum(1).mean()

    def cross_entropy_t(self, logits, targets, mask, reduction='batchmean'):

        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)

        log_pos = torch.gather(logsoftmax, 1, (targets*mask).long())
        return - (log_pos * mask).sum() / mask.sum()

    def compute_train(self,model,sample):
        batch_size, win, ndim = sample.shape
        list_vec = []
        for i in np.arange(0,batch_size,64):
            list_vec.append(model(sample[i:i+64].cuda(),step=0))
            outputs_samples = torch.cat(list_vec, 0)
        logits = model.fc(outputs_samples)
        return logits

    def run_task(self, task_dic, model,test_file, first):
        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model,self.method,test_file, first) # choose the update methods
        # Extract support and query
        y_s = task_dic['y_s']  # n_task*?*?
        z_s, z_q = task_dic['z_s'], task_dic['z_q']
        min_len = task_dic['MFL']
        mean_pos_len = task_dic['mean_pos_len']
        z_t,y_t = task_dic['z_t'], task_dic['y_t']
        mask = task_dic['mask']
        
        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        sub_train = z_t.to(self.device)
        mask = mask.to(self.device)

        y_s = y_s.long().to(self.device) 
        y_t = y_t.long().to(self.device) 
      
        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s) # lambda
        print('tim_builder.loss_weights',tim_builder.loss_weights[0])
        print('self.number_task ',self.number_tasks)
        tim_builder.init_weights(support=support, y_s=y_s, query=query, sub_train=sub_train, y_t=y_t) # init W
        tim_builder.compute_FB_param(query)
        # Run adaptation
        tim_builder.run_adaptation(support=support, query=query, y_s=y_s,min_len=mean_pos_len, sub_train=sub_train,y_t=y_t,mask_t=mask) # update
        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return logs

    def get_tim_builder(self,model,method,test_file,first):
        # Initialize TIM classifier builder
        tim_info = {'model': model,'test_file': test_file,'first': first}
        if method == 'tim_adm':
            tim_builder = TIM_ADM(**tim_info)
        elif method == 'tim_gd':
            tim_builder = TIM_GD(**tim_info)
        elif method == 'baseline':
            tim_builder = TIM(**tim_info)
        else:
            raise ValueError("Method must be in ['tim_gd', 'tim_adm', 'baseline']")
        return tim_builder

    def get_loaders(self, hdf_eval,conf):
        # First, get loaders
        loaders_dic = {}
        gen_eval = Datagen_test(hdf_eval,conf)
        
        X_pos_1, X_pos_2, X_pos_3, X_pos_4, X_pos_5,X_neg_1, X_neg_2, X_neg_3, X_neg_4, X_neg_5, X_query = gen_eval.generate_eval()

        save_dict = {}
        save_dict['x_pos_1'] = X_pos_1
        save_dict['x_pos_2'] = X_pos_2
        save_dict['x_pos_3'] = X_pos_3
        save_dict['x_pos_4'] = X_pos_4
        save_dict['x_pos_5'] = X_pos_5
       
        save_dict.update({'x_query': X_query})
        self.number_tasks = X_query.shape[0]

        X_pos_1 = torch.tensor(X_pos_1)
        Y_pos_1 = torch.LongTensor(np.ones(X_pos_1.shape[0]))

        X_pos_2 = torch.tensor(X_pos_2)
        Y_pos_2 = torch.LongTensor(np.ones(X_pos_2.shape[0]))

        X_pos_3 = torch.tensor(X_pos_3)
        Y_pos_3 = torch.LongTensor(np.ones(X_pos_3.shape[0]))

        X_pos_4 = torch.tensor(X_pos_4)
        Y_pos_4 = torch.LongTensor(np.ones(X_pos_4.shape[0]))

        X_pos_5 = torch.tensor(X_pos_5)
        Y_pos_5 = torch.LongTensor(np.ones(X_pos_5.shape[0]))

        X_neg_1 = torch.tensor(X_neg_1)
        Y_neg_1 = torch.LongTensor(np.zeros(X_neg_1.shape[0]))

        X_neg_2 = torch.tensor(X_neg_2)
        Y_neg_2 = torch.LongTensor(np.zeros(X_neg_2.shape[0]))

        X_neg_3 = torch.tensor(X_neg_3)
        Y_neg_3 = torch.LongTensor(np.zeros(X_neg_3.shape[0]))

        X_neg_4 = torch.tensor(X_neg_4)
        Y_neg_4 = torch.LongTensor(np.zeros(X_neg_4.shape[0]))

        X_neg_5 = torch.tensor(X_neg_5)
        Y_neg_5 = torch.LongTensor(np.zeros(X_neg_5.shape[0]))

        X_query = torch.tensor(X_query)
        Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

        # for i in range(5):
        #     self.writer.add_img("X_pos",merge_image,0)
        
        query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
        q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False) 
        loaders_dic['query'] = q_loader

        pos_dataset_1 = torch.utils.data.TensorDataset(X_pos_1, Y_pos_1)
        pos_dataset_2 = torch.utils.data.TensorDataset(X_pos_2, Y_pos_2)
        pos_dataset_3 = torch.utils.data.TensorDataset(X_pos_3, Y_pos_3)
        pos_dataset_4 = torch.utils.data.TensorDataset(X_pos_4, Y_pos_4)
        pos_dataset_5 = torch.utils.data.TensorDataset(X_pos_5, Y_pos_5)
        neg_dataset_1 = torch.utils.data.TensorDataset(X_neg_1, Y_neg_1)
        neg_dataset_2 = torch.utils.data.TensorDataset(X_neg_2, Y_neg_2)
        neg_dataset_3 = torch.utils.data.TensorDataset(X_neg_3, Y_neg_3)
        neg_dataset_4 = torch.utils.data.TensorDataset(X_neg_4, Y_neg_4)
        neg_dataset_5 = torch.utils.data.TensorDataset(X_neg_5, Y_neg_5)

        pos_loader_1 = torch.utils.data.DataLoader(dataset=pos_dataset_1,batch_sampler=None, batch_size=50,shuffle=False)
        pos_loader_2 = torch.utils.data.DataLoader(dataset=pos_dataset_2,batch_sampler=None, batch_size=50,shuffle=False)
        pos_loader_3 = torch.utils.data.DataLoader(dataset=pos_dataset_3,batch_sampler=None, batch_size=50,shuffle=False)
        pos_loader_4 = torch.utils.data.DataLoader(dataset=pos_dataset_4,batch_sampler=None, batch_size=50,shuffle=False)
        pos_loader_5 = torch.utils.data.DataLoader(dataset=pos_dataset_5,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_1 = torch.utils.data.DataLoader(dataset=neg_dataset_1,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_2 = torch.utils.data.DataLoader(dataset=neg_dataset_2,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_3 = torch.utils.data.DataLoader(dataset=neg_dataset_3,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_4 = torch.utils.data.DataLoader(dataset=neg_dataset_4,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_5 = torch.utils.data.DataLoader(dataset=neg_dataset_5,batch_sampler=None, batch_size=50,shuffle=False)
        
        loaders_dic.update({'pos_loader_1': pos_loader_1})
        loaders_dic.update({'pos_loader_2': pos_loader_2})
        loaders_dic.update({'pos_loader_3': pos_loader_3})
        loaders_dic.update({'pos_loader_4': pos_loader_4})
        loaders_dic.update({'pos_loader_5': pos_loader_5})
        loaders_dic.update({'neg_loader_1': neg_loader_1})
        loaders_dic.update({'neg_loader_2': neg_loader_2})
        loaders_dic.update({'neg_loader_3': neg_loader_3})
        loaders_dic.update({'neg_loader_4': neg_loader_4})
        loaders_dic.update({'neg_loader_5': neg_loader_5})

        return loaders_dic,save_dict

    def extract_features(self, model, model_path, model_tag, used_set, fresh_start, loaders_dic,test_student=0):
        # Load features from memory if previously saved ...
        save_dir = os.path.join(model_path, model_tag, used_set)
        filepath = os.path.join(save_dir, 'output.plk')
        if os.path.isfile(filepath) and (not fresh_start):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic
        
        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            os.makedirs(save_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            all_features = []
            all_labels = []
            # print("===> Query feature extraction")
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['query'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs)
                all_labels.append(labels)
            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            extracted_features_dic = {'query_features': all_features,
                                      'query_labels': all_labels}
            all_features = []
            all_labels = []

            # print("===> Pos feature extraction")
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_1'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)

            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_2'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)

            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_3'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)

            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_4'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)
            
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_5'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)
            # all_features = torch.cat(all_features, 0)
            # all_labels = torch.cat(all_labels, 0)
            extracted_features_dic.update({'pos_features': all_features,
                                      'pos_labels': all_labels})

            all_features = []
            all_labels = []

            # print("===> Neg feature extraction")
            bad_cnt = 0
            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_1'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_2'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_3'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_4'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_5'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            for i in range(bad_cnt):
                index = i%(5-bad_cnt)
                all_features.append(all_features[index])
                all_labels.append(all_labels[index])

            # all_features = torch.cat(all_features, 0)
            # all_labels = torch.cat(all_labels, 0)
            extracted_features_dic.update({'neg_features': all_features,
                                      'neg_labels': all_labels})
            extra_neg = self.get_extra_neg(model,extracted_features_dic['pos_features'], extracted_features_dic['query_features'])
            extracted_features_dic.update({'extra_neg':extra_neg})

        print(" ==> Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic

    def get_extra_neg(self,model,pos_features,query_features):
        model.eval()
        # print("===> extra Neg feature extraction")
        with torch.no_grad():
            list_pos_vec = []
            for i in warp_tqdm(range(len(pos_features)), True):

                pos_input = torch.zeros_like(query_features[:1])
                start = max((query_features.shape[1]-pos_features[i].shape[0])//2,0)
                end = min(start+pos_features[i].shape[0],query_features.shape[1])
                pos_input[0,start:end] = pos_features[i][:(end-start)]
                
                pos_outputs = model.forward_encoder_test(pos_input.to(self.device))
                list_pos_vec.append(pos_outputs[:,start:end])
            
            pos_w = torch.cat(list_pos_vec,1).mean(1,keepdim=True)
            pos_w = F.normalize(pos_w,dim=2)

            list_neg = []
            for i in warp_tqdm(range(query_features.shape[0]),True):
                que_outputs = model.forward_encoder_test(query_features[i:(i+1)].to(self.device))
                _,neg_idx = (que_outputs*pos_w).sum(2).sort()
                list_neg.append(query_features[i, neg_idx[0,50:60].cpu(),:])
            torch_neg = torch.cat(list_neg,0)
        return torch_neg
    
    def get_task(self, extracted_features_dic, conf, model,loaders_dic):

        """
        inputs:
            extracted_features_dic : Dictionnary containing all extracted features and labels
            shot : Number of support shot per class
            n_ways : Number of ways for the task

        returns :
            task : Dictionnary : z_support : torch.tensor of shape [n_ways * shot, feature_dim]
                                 z_query : torch.tensor of shape [n_ways * query_shot, feature_dim]
                                 y_support : torch.tensor of shape [n_ways * shot]
                                 y_query : torch.tensor of shape [n_ways * query_shot]
        """
        query_features = extracted_features_dic['query_features']
        pos_features = extracted_features_dic['pos_features']
        neg_features = extracted_features_dic['neg_features']
        extra_neg = extracted_features_dic['extra_neg']

        query_samples = []
        query_samples.append(query_features)
        z_query = torch.cat(query_samples,0)

        support_samples = []
        
        y_list = []

        list_len_pos =[fea.shape[0] for fea in pos_features]
        list_len_neg =[fea.shape[0] for fea in neg_features]

        mean_pos_len = sum(list_len_pos)/len(list_len_pos)
        print('mean_pos_len:%s'%mean_pos_len)
        med_filter_len = min(list_len_pos)
        print('min_pos_len:%s'%med_filter_len)
        n_frame = 431
        max_seg_len = n_frame//2
        # print("====> Build features")
        for i in warp_tqdm(range(128),True):
            list_X = []
            list_Y = []
            len_cnt = 0
            while True:
                neg_id = random.choice(range(len(neg_features)))

                if sum(list_len_neg)<10 and i%2==1 and neg_features[neg_id].shape[0]<5:
                    neg_len = random.choice(range(10,50))
                    start = random.choice(range(extra_neg.shape[0]-neg_len))
                    list_X.append(extra_neg[start:start+neg_len])
                    list_Y.append(torch.zeros(neg_len))
                    len_cnt += neg_len
                else:
                    if neg_features[neg_id].shape[0] > max_seg_len:
                        start = random.choice(range(neg_features[neg_id].shape[0]- max_seg_len))
                        end = start+ max_seg_len
                    else:
                        start = 0
                        end = neg_features[neg_id].shape[0]
                    list_X.append(neg_features[neg_id][start:end])
                    list_Y.append(torch.zeros(end-start))
                    len_cnt +=end -start

                pos_id = random.choice(range(len(pos_features)))
                if pos_features[pos_id].shape[0] >max_seg_len:
                    start = random.choice(range(pos_features[pos_id].shape[0]-max_seg_len))
                    end = start+max_seg_len
                else:
                    start =0
                    end = pos_features[pos_id].shape[0]

                if len_cnt+end-start <=n_frame:
                    list_X.append(pos_features[pos_id][start:end])
                    list_Y.append(torch.ones(end-start))
                    len_cnt += end-start
                
                if len_cnt>n_frame:
                    break

            support_samples.append(torch.cat(list_X,0)[:n_frame])
            y_list.append(torch.cat(list_Y,0)[:n_frame])
            
        z_support = torch.stack(support_samples, 0)
        y_support = torch.stack(y_list, 0)
        #reload sub_train_datasets
        try:
            sub_train_datasets  =torch.load(conf.eval.trainDatasets)
            z_sub_train = sub_train_datasets['z_sub_train']
            y_sub_train = sub_train_datasets['y_sub_train']
          
        except:
            gen_sub_train = Datagen_train_select(conf)
            z_sub_train,y_sub_train = gen_sub_train.generate_eval()
            z_sub_train = torch.tensor(z_sub_train).type_as(z_support)
            y_sub_train = torch.tensor(y_sub_train).type_as(y_support)
           
            sub_train_datasets ={
                'z_sub_train': z_sub_train,
                'y_sub_train': y_sub_train
            }
            torch.save(sub_train_datasets,conf.eval.trainDatasets)
        mask = torch.where(y_sub_train>0,1,0)  
        z_train_dataset = torch.utils.data.TensorDataset(z_sub_train,y_sub_train)
        z_train_dataloader = torch.utils.data.DataLoader(dataset=z_train_dataset,batch_sampler=None,batch_size=64,num_workers=0,shuffle=True)
        loaders_dic['z_trainloader'] = z_train_dataloader
        
        z_sub_train = torch.cat((z_sub_train,z_support),0)
        y_sub_train = torch.cat((y_sub_train,y_support-1),0)
        
        mask = torch.cat((mask,y_support),0)
        
        task = {'z_s': z_support.contiguous(), 'y_s': y_support.contiguous(),
                'z_t': z_sub_train.contiguous(), 'y_t': y_sub_train.contiguous(),'mask':mask.contiguous(),
                'z_q': z_query.contiguous(),'MFL':med_filter_len, 'mean_pos_len':mean_pos_len}
        return task,loaders_dic  
    
    def generate_tasks(self, extracted_features_dic,conf,model,loaders_dic):
        """
        inputs:
            extracted_features_dic :
            shot : Number of support shot per class
            number_tasks : Number of tasks to generate

        returns :
            merged_task : { z_support : torch.tensor of shape [number_tasks, n_ways * shot, feature_dim]
                            z_query : torch.tensor of shape [number_tasks, n_ways * query_shot, feature_dim]
                            y_support : torch.tensor of shape [number_tasks, n_ways * shot]
                            y_query : torch.tensor of shape [number_tasks, n_ways * query_shot] }
        """
        return self.get_task(extracted_features_dic,conf,model,loaders_dic)

