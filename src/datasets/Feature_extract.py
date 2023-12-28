import os
import librosa
import h5py
import pandas as pd
import numpy as np
from scipy import signal
from glob import glob
from itertools import chain
from torchaudio import functional as F
import torchaudio.transforms as T   
import torch

pd.options.mode.chained_assignment = None

def create_trainDataset(df_pos,pcen,glob_cls_name,hf,seg_len,hop_seg,fps):

    '''Chunk the time-frequecy representation to segment length and store in h5py dataset

    Args:
        -df_pos : dataframe
        -log_mel_spec : log mel spectrogram
        -glob_cls_name: Name of the class used in audio files where only one class is present
        -file_name : Name of the csv file
        -hf: h5py object
        -seg_len : fixed segment length
        -fps: frame per second
    Out:
        - label_list: list of labels for the extracted mel patches'''

    label_list = []
    if len(hf['features'][:]) == 0:
        file_index = 0  
    else:
        file_index = len(hf['features'][:])

    start_time,end_time = time_2_frame(df_pos, fps) #

    'For csv files with a column name Call, pick up the global class name'
    if 'CALL' in df_pos.columns:
        cls_list = [glob_cls_name] * len(start_time)
    else:
        cls_list = [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, row in df_pos.iterrows()]
        cls_list = list(chain.from_iterable(cls_list))

    assert len(start_time) == len(end_time)
    assert len(cls_list) == len(start_time)

    nframe_label = ['0']*pcen.shape[0]
    for index in range(len(start_time)):
        str_ind = start_time[index]
        end_ind = end_time[index]
        label = cls_list[index] 
        for ind in range(str_ind,end_ind):
            nframe_label[ind]=label
            
        # 'Extract segment and move forward with hop_seg'
    cur_time = 0
    while cur_time+seg_len<=pcen.shape[0]: #
        pcen_patch = pcen[int(cur_time):int(cur_time+seg_len)]
        hf['features'].resize((file_index+1,pcen_patch.shape[0],pcen_patch.shape[1]))
        hf['features'][file_index] = pcen_patch
        file_index +=1
       
        # The overlap sound events are not trained
        middle_idx = 0
        label_path = nframe_label[int(cur_time):int(cur_time+seg_len)]
        if cur_time and nframe_label[int(cur_time)-1] !='0' and nframe_label[int(cur_time)+1] !='0':
            try:
                middle_idx = label_path.index('0')
            except:
                middle_idx = seg_len
        for idx in range(0,middle_idx):
            label_list.append('-1')
            
        end_idx = seg_len
        if cur_time+seg_len< pcen.shape[0] and nframe_label[int(cur_time+seg_len)-2] !='0' \
            and nframe_label[int(cur_time+seg_len)] !='0':
            temp_label = label_path[middle_idx:]
            temp_label.reverse()
            try:
                end_idx = seg_len-temp_label.index('0')
            except:
                end_idx = seg_len
        for idx in range(middle_idx,end_idx):
            label_list.append(label_path[idx])
        for idx in range(end_idx,seg_len):
            label_list.append('-1')
        cur_time = cur_time + hop_seg
    assert len(label_list) == (cur_time//hop_seg)*seg_len
    print('Total files created:{}'.format(file_index))
    return label_list

def create_evalDataset(pcen, conf, seg_len, hop_seg, 
                       file_name, df_eval, hf, fps, num_extract_eval= 0):
    # ----------------------------------------------------------------------------------------
    Q_list = df_eval['Q'].to_numpy() # Q column
    
    start_time,end_time = time_2_frame(df_eval, fps)
    index_sup = np.where(Q_list == 'POS')[0][:conf.features.n_shot] 
    
    strt_indx_query = end_time[index_sup[-1]] # The start point of query, which is the end point of the last call in support
    end_idx_neg = pcen.shape[0] - 1 
    hf['start_index_query'][:] = strt_indx_query
    
    print("Creating Positive dataset from {}".format(file_name))
    idx_pos = 0
    for index in index_sup:
        str_ind = max(0,int(start_time[index]))
        end_ind = int(end_time[index])

        patch_pos = pcen[int(str_ind):int(end_ind)]

        hf.create_dataset('feat_pos_%s'%idx_pos,shape=(0,patch_pos.shape[0],patch_pos.shape[1]),maxshape=(None,patch_pos.shape[0],patch_pos.shape[1]))
        hf['feat_pos_%s'%idx_pos].resize((0+1,patch_pos.shape[0],patch_pos.shape[1]))
        hf['feat_pos_%s'%idx_pos][0]=patch_pos
        idx_pos +=1
    print('index_Pos: ', idx_pos)
    print("Creating Negative dataset from {}".format(file_name))
    start_time,end_time = time_2_frame(df_eval,fps, jitter=0)
    idx_neg = 0
    str_ind = 0
    
    for i in range(0,index_sup.shape[0]):
        index = index_sup[i]
        end_ind = max(0,int(start_time[index]))
        patch_pos = pcen[int(str_ind):int(end_ind)]
        hf.create_dataset('feat_neg_%s'%idx_neg,shape=(0,patch_pos.shape[0],patch_pos.shape[1]),maxshape=(None,patch_pos.shape[0],patch_pos.shape[1]))
        hf['feat_neg_%s'%idx_neg].resize((0+1,patch_pos.shape[0],patch_pos.shape[1]))
        hf['feat_neg_%s'%idx_neg][0] = patch_pos
        idx_neg +=1
        str_ind = int(end_time[index])

    print("Creating query dataset from {}".format(file_name))
    idx_query = 0
    hop_query = 0
    while end_idx_neg - (strt_indx_query+hop_query) > seg_len:
        patch_query = pcen[int(strt_indx_query+hop_query):int(strt_indx_query+hop_query+seg_len)] 
        hf['feat_query'].resize((idx_query+1,patch_query.shape[0],patch_query.shape[1]))
        hf['feat_query'][idx_query]=patch_query
        idx_query +=1
        hop_query += hop_seg
    print("index_Query", idx_query)
    last_patch_query = pcen[end_idx_neg-seg_len:end_idx_neg]
    hf['feat_query'].resize((idx_query+1,last_patch_query.shape[0],last_patch_query.shape[1]))
    hf['feat_query'][idx_query] = last_patch_query
    num_extract_eval += len(hf['feat_query'])
    hf.close()
    return num_extract_eval
  
def create_testDataset(pcen, conf, seg_len, hop_seg, 
                       file_name, df_eval, hf, fps, num_extract_eval= 0):
    # ----------------------------------------------------------------------------------------
    Q_list = df_eval['Q'].to_numpy() # Q column
    
    start_time,end_time = time_2_frame(df_eval, fps)
    index_sup = np.where(Q_list == 'POS')[0][:conf.feature.n_shot] 
    unk_index = np.where(Q_list == 'UNK')[0]
    
    if len(unk_index)>0:
        unk_sup = np.stack([start_time, end_time]).T
    else:
        unk_sup = np.empty(shape=(0,2))
                
    strt_indx_query = end_time[index_sup[-1]] # The start point of query, which is the end point of the last call in support
    end_idx_neg = pcen.shape[0] - 1 
    hf['start_index_query'][:] = strt_indx_query
    
    print("Creating Positive dataset from {}".format(file_name))
    idx_pos = 0
    for index in index_sup:
        str_ind = max(0,int(start_time[index]))
        end_ind = int(end_time[index])

        patch_pos = pcen[int(str_ind):int(end_ind)]

        hf.create_dataset('feat_pos_%s'%idx_pos,shape=(0,patch_pos.shape[0],patch_pos.shape[1]),maxshape=(None,patch_pos.shape[0],patch_pos.shape[1]))
        hf['feat_pos_%s'%idx_pos].resize((0+1,patch_pos.shape[0],patch_pos.shape[1]))
        hf['feat_pos_%s'%idx_pos][0]=patch_pos
        idx_pos +=1
    print('index_Pos: ', idx_pos)
    print("Creating Negative dataset from {}".format(file_name))
    start_time,end_time = time_2_frame(df_eval,fps, jitter=0)
    idx_neg = 0
    str_ind = 0
    
    for i in range(0,index_sup.shape[0]):
        index = index_sup[i]
        end_ind = max(0,int(start_time[index]))
        
        # use UNK augment N_i
        if len(unk_sup)>0:
            unk_index = (unk_sup[:,0]>=str_ind) & (unk_sup[:,1]<=end_ind)
            patch_unk = unk_sup[unk_index][np.argsort(unk_sup[unk_index][:,0], axis=0)]
            if len(patch_unk)>0:
                patch_pos_list = []     
                for unk in patch_unk:
                    patch_end = unk[0]
                    patch_pos_list.append(pcen[int(str_ind):int(patch_end)])
                    str_ind = unk[1]
                patch_pos_list.append(pcen[int(str_ind):int(end_ind)])
                patch_pos = np.concatenate(patch_pos_list, axis=0) 
            else:
                patch_pos = pcen[int(str_ind):int(end_ind)]
        else:
            patch_pos = pcen[int(str_ind):int(end_ind)]      
        patch_pos = pcen[int(str_ind):int(end_ind)]
        hf.create_dataset('feat_neg_%s'%idx_neg,shape=(0,patch_pos.shape[0],patch_pos.shape[1]),
                          maxshape=(None,patch_pos.shape[0],patch_pos.shape[1]))
        hf['feat_neg_%s'%idx_neg].resize((0+1,patch_pos.shape[0],patch_pos.shape[1]))
        hf['feat_neg_%s'%idx_neg][0] = patch_pos
        idx_neg +=1
        str_ind = int(end_time[index])

    print("Creating query dataset from {}".format(file_name))
    idx_query = 0
    hop_query = 0
    while end_idx_neg - (strt_indx_query+hop_query) > seg_len:
        patch_query = pcen[int(strt_indx_query+hop_query):int(strt_indx_query+hop_query+seg_len)] 
        hf['feat_query'].resize((idx_query+1,patch_query.shape[0],patch_query.shape[1]))
        hf['feat_query'][idx_query]=patch_query
        idx_query +=1
        hop_query += hop_seg
    print("index_Query", idx_query)
    last_patch_query = pcen[end_idx_neg-seg_len:end_idx_neg]
    hf['feat_query'].resize((idx_query+1,last_patch_query.shape[0],last_patch_query.shape[1]))
    hf['feat_query'][idx_query] = last_patch_query
    num_extract_eval += len(hf['feat_query'])
    hf.close()
    
    return num_extract_eval 
    
class Feature_Extractor():

       def __init__(self, conf):
           self.sr =conf.features.sr
           self.n_fft = conf.features.n_fft
           self.hop = conf.features.hop_mel
           self.n_mels = conf.features.n_mels
           self.fmax = conf.features.fmax
           
       def extract_feature(self,audio):

           mel_spec = librosa.feature.melspectrogram(audio,sr=self.sr, n_fft=self.n_fft,
                                                     hop_length=self.hop,n_mels=self.n_mels,fmax=self.fmax)
           pcen = librosa.core.pcen(mel_spec,sr=22050)
           pcen = pcen.astype(np.float32)
           return pcen

def extract_feature(audio_path,feat_extractor,conf):

    y, _ = librosa.load(audio_path,sr=conf.features.sr)

    'Scaling audio as per suggestion in librosa documentation'
    y = y * (2**32) # energy amplification
    pcen = feat_extractor.extract_feature(y)
    
    return pcen.T

def time_2_frame(df,fps, jitter=0.025):
    'Margin of 25 ms around the onset and offsets'

    df.loc[:,'Starttime'] = df['Starttime'] - jitter
    df.loc[:,'Endtime'] = df['Endtime'] + jitter

    'Converting time to frames'

    start_time = [int(np.floor(start * fps)) for start in df['Starttime']] # fps That is, how many frames are there in 1s, start*fps can get the start frame

    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    return start_time,end_time


def feature_transform(conf=None,mode=None):
    """The PCEN feature extraction and sliding window segmentation for Dev/Eval set

    Args:
        conf (yaml, optional): The config in features of config.yaml. 
        mode (str, optional): train/ eval/ test.
    Returns:
        int: The total numbers of query segments
    """
    label_tr = []
    pcen_extractor = Feature_Extractor(conf)

    fps =  conf.features.sr / conf.features.hop_mel  # 22050/256=86
    'Converting fixed segment legnth to frames'
    # print('fps ',fps)
    seg_len = int(round(conf.features.seg_len * fps)) # 86*0.200= 17
    # print('seg_len ',seg_len)
    hop_seg = int(round(conf.features.hop_seg * fps)) # 86*0.05= 4
    # print('hop_seg ',hop_seg)
    extension = "*.csv"

    if mode == 'train':
        print("=== Processing training set ===")
        meta_path = conf.path.train_dir # training set path
        all_csv_files = [file
                         for path_dir, _, _ in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, extension))] #  get all csv path
        hdf_tr = os.path.join(conf.path.feat_train,'Mel_train.h5') # mel save path
        hf = h5py.File(hdf_tr,'w')
        hf.create_dataset('features', shape=(0, seg_len, conf.features.n_mels),
                          maxshape=(None, seg_len, conf.features.n_mels))
        num_extract = 0
        for file in all_csv_files:
            
            split_list = file.split('/')
            glob_cls_name = split_list[split_list.index('Training_Set') + 1] 
            df = pd.read_csv(file, header=0, index_col=False)
            audio_path = file.replace('csv', 'wav')
            print("Processing file name {}".format(audio_path))
            pcen = extract_feature(audio_path, pcen_extractor,conf)
            df_pos = df[(df == 'POS').any(axis=1)] # find the POS annotation
            label_list = create_trainDataset(df_pos, pcen, glob_cls_name, hf, seg_len, hop_seg, fps)
            label_tr.append(label_list)
            # break
        print(" Feature extraction for training set complete")
        num_extract = len(hf['features'])
        flat_list = [item for sublist in label_tr for item in sublist] # merge the sublist
        hf.create_dataset('labels', data=[s.encode() for s in flat_list], dtype='S20')
        data_shape = hf['features'].shape
        hf.close()
        return num_extract,data_shape

    elif mode=='eval':

        print("=== Processing Validation set ===")

        meta_path = conf.path.eval_dir
        num_extract_eval = 0
        
        all_csv_files = [file
                         for path_dir, _, _ in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, extension))]

        for file in all_csv_files:
            split_list = file.split('/')
            name = str(split_list[-1].split('.')[0])
    
            feat_name = name + '.h5'
            audio_path = file.replace('csv', 'wav')
            hdf_eval = os.path.join(conf.path.feat_eval,feat_name)
            hf = h5py.File(hdf_eval,'w')
            
            hf.create_dataset('feat_pos', shape=(0, seg_len, conf.features.n_mels),
                                maxshape= (None, seg_len, conf.features.n_mels))
            hf.create_dataset('feat_query',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('feat_neg',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('start_index_query',shape=(1,),maxshape=(None))

            "In case you want to use the statistics of each file to normalize"
            hf.create_dataset('mean_global',shape=(1,),maxshape=(None))
            hf.create_dataset('std_dev_global',shape=(1,),maxshape=(None))
            df_eval = pd.read_csv(file, header=0, index_col=False)
            
            pcen = extract_feature(audio_path, pcen_extractor, conf) # pcen feature extraction
            hf['mean_global'][:] = np.mean(pcen)
            hf['std_dev_global'][:] = np.std(pcen) 
            
            num_extract_eval = create_evalDataset(pcen, conf, seg_len, hop_seg, file, df_eval, hf, fps, num_extract_eval)
        return num_extract_eval
    
    elif mode=='test':

        print("=== Processing Test set ===")

        meta_path = conf.path.test_dir

        all_csv_files = [file
                         for path_dir, _, _ in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, extension))]

        num_extract_eval = 0
        
        for file in all_csv_files:
            
            split_list = file.split('/')
            name = str(split_list[-1].split('.')[0])
            feat_name = name + '.h5'
            audio_path = file.replace('csv', 'wav')
    
            hdf_eval = os.path.join(conf.path.feat_test,feat_name)
            hf = h5py.File(hdf_eval,'w')

            hf.create_dataset('feat_pos', shape=(0, seg_len, conf.features.n_mels),
                              maxshape= (None, seg_len, conf.features.n_mels))
            hf.create_dataset('feat_query',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('feat_neg',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('start_index_query',shape=(1,),maxshape=(None))
            hf.create_dataset('mean_global',shape=(1,),maxshape=(None))
            hf.create_dataset('std_dev_global',shape=(1,),maxshape=(None))
            df_eval = pd.read_csv(file, header=0, index_col=False)

            pcen = extract_feature(audio_path, pcen_extractor,conf) # pcen feature extraction
            hf['mean_global'][:] = np.mean(pcen)
            hf['std_dev_global'][:] =  np.std(pcen)         

            num_extract_eval = create_testDataset(pcen, conf, seg_len, hop_seg, file, df_eval, hf, fps, num_extract_eval)
        return num_extract_eval









