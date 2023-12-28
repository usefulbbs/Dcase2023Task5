
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import warnings
warnings.filterwarnings("ignore")

def class_to_int(label_array,class_set):

    '''  Convert class label to integer

    Args:
    -label_array: label array
    -class_set: unique classes in label_array

    Out:
    -y: label to index values
    '''
    class_set.remove('-1')
    class_set.remove('0')
    label2indx = {label:index+1 for index,label in enumerate(class_set)}
    label2indx['-1'] = -1
    label2indx['0'] = 0
    y = np.array([label2indx[label] for label in label_array])
    return y

def list_to_string(list_value):
    list_str = [str(i) for i in list_value]
    return '_'.join(list_str)

def balance_class_distribution(X,Y):

    '''  Class balancing through Random oversampling
    Args:
    -X: Feature
    -Y: labels

    Out:
    -X_new: Feature after oversampling
    -Y_new: Oversampled label list
    '''
    nframe_win = 431
    x_index  = []
    fake_Y = []
    dict_key = {}
    dict_posX = {}
    X_pos = np.zeros_like(X)
    for i in range(len(X)):
        label_win = Y[i*nframe_win:(i+1)*nframe_win]
        list_y = np.unique(label_win).tolist()
        list_y.sort()
        if 0 in list_y: list_y.remove(0)
        if -1 in list_y: list_y.remove(-1)
        
        if len(list_y) == 0:
            continue 
        elif len(list_y) == 1:
            class_id = list_y[0]
        else:
            key = list_to_string(list_y)
            list_key = key.split('_')
            if key not in dict_key.keys():
                dict_key[key] = 0
            else: 
                dict_key[key] = (dict_key[key]+1)%len(list_key)
            class_id = int(list_key[dict_key[key]])
        fake_Y.append(class_id)
        x_index.append([i])
        
        repeat_label = label_win.reshape(-1,1).repeat(128,1)
        masked_X = np.where(repeat_label==class_id, X[i], 0)
        if class_id not in dict_posX.keys():
            X_pos[i] = masked_X
        else:
            X_pos[i] = dict_posX[class_id]
        dict_posX[class_id] = masked_X
    fake_Y = np.array(fake_Y,dtype=np.int64)
    
    all_X = np.concatenate((X,X_pos),axis=1)
    ros = RandomOverSampler(random_state=42)
    x_unifm, y_unifm = ros.fit_resample(x_index, fake_Y)
    unifm_index = [index_new[0] for index_new in x_unifm]
    X_new = np.array([all_X[index] for index in unifm_index])
    Y_new = np.array([Y[idx*nframe_win:(idx+1)*nframe_win] for idx in unifm_index])
    return X_new,Y_new, y_unifm


def balance_class_distribution_sub(X,Y):

    '''  Class balancing through Random oversampling
    Args:
    -X: Feature
    -Y: labels

    Out:
    -X_new: Feature after oversampling
    -Y_new: Oversampled label list
    '''
    nframe_win = 431
    x_index  = []
    fake_Y = []
    dict_key = {}
    for i in range(len(X)):
        label_win = Y[i*nframe_win:(i+1)*nframe_win]
        list_y = np.unique(label_win).tolist()
        list_y.sort()
        if 0 in list_y: list_y.remove(0)
        if -1 in list_y: list_y.remove(-1)
        if len(list_y)==0:
            continue 
        elif len(list_y)==1:
            class_id = list_y[0]
        else:
            key = list_to_string(list_y)
            list_key = key.split('_')
            if key not in dict_key.keys():
                dict_key[key]=0
            else:
                dict_key[key]=(dict_key[key]+1)%len(list_key)

            class_id = int(list_key[dict_key[key]])
        fake_Y.append(class_id)
        x_index.append([i])
    fake_Y = np.array(fake_Y,dtype=np.int64)
    sampling_strategy = {}
    for i in range(1,20):
        sampling_strategy[i]=24
    ros = RandomUnderSampler(random_state=42,sampling_strategy=sampling_strategy)
    x_unifm, y_unifm = ros.fit_resample(x_index, fake_Y)
    unifm_index = [index_new[0] for index_new in x_unifm]

    X_new = np.array([X[index] for index in unifm_index])
    Y_new = np.array([Y[idx*nframe_win:(idx+1)*nframe_win] for idx in unifm_index])

    return X_new,Y_new,y_unifm

def norm_params(X):

    '''  Normalize features
        Args:
        - X : Features

        Out:
        - mean : Mean of the feature set
        - std: Standard deviation of the feature set
        '''
    mean = np.mean(X)
    std = np.std(X)
    return mean, std

class Datagen(object):

    def __init__(self, conf, is_test=False):
        if not is_test:
            hdf_path = os.path.join(conf.path.feat_train, 'Mel_train.h5')
            self.work_space_path = conf.path.work_path
            hdf_train = h5py.File(hdf_path, 'r+')
            self.x = hdf_train['features'][:]
            self.labels = [s.decode() for s in hdf_train['labels'][:]]
            self.nframe = self.x.shape[1]
            self.class_set = set(self.labels)    
            self.y = class_to_int(self.labels,self.class_set) # str to int
            
            self.x,self.y,self.y_unifm = balance_class_distribution(self.x,self.y)
            self.y2 = np.zeros_like(self.y)
            for i in range(self.y_unifm.shape[0]):
                self.y2[i] = np.where(self.y[i]==self.y_unifm[i],1,0)
                
            array_train = np.arange(len(self.x))
            array_train = np.arange(len(self.x))
            _,_,_,_,_,_,train_array,valid_array = train_test_split(self.x,self.y,self.y2,array_train,random_state=42,stratify=self.y_unifm)
            self.train_index = train_array
            self.valid_index = valid_array
            self.mean,self.std = norm_params(self.x[train_array,:self.nframe])
            with open("%s/src/mean_var"%conf.path.work_path,'w') as fw:
                fw.writelines('%s %s %s'%(self.mean,self.std,self.nframe))
        else:
            with open("%s/src/mean_var"%conf.path.work_path,'r') as fr:
                list_line = fr.readlines()[0].strip().split()
                self.mean = float(list_line[0])
                self.std = float(list_line[1])
                self.nframe = int(list_line[2])
    
    def feature_scale(self,X):
        X[:,:self.nframe] = (X[:,:self.nframe]-self.mean)/self.std
        X[:,self.nframe:] = np.where(X[:,self.nframe:]!=0,(X[:,self.nframe:]-self.mean)/self.std,0)
        return X
        
    def generate_train(self):

        ''' Returns normalized training and validation features.
        Args:
        -conf - Configuration object
        Out:
        - X_train: Training features
        - X_val: Validation features
        - Y_train: Training labels
        - Y_Val: Validation labels
        - Y2_train: VAD training labels
        - Y2_Val: VAD validation labels
        '''
        train_array = sorted(self.train_index)
        valid_array = sorted(self.valid_index)
        X_train = self.x[train_array]
        Y_train = self.y[train_array]
        Y2_train = self.y2[train_array]
        
        X_val = self.x[valid_array]
        Y_val = self.y[valid_array]
        Y2_val = self.y2[valid_array]

        X_train = self.feature_scale(X_train)
        X_val = self.feature_scale(X_val)
        return X_train,Y_train,Y2_train,X_val,Y_val,Y2_val

class Datagen_train_select(Datagen):
    def __init__(self,conf):
        super(Datagen_train_select, self).__init__(conf= conf,is_test=True)
        hdf_path = os.path.join(conf.path.feat_train,'Mel_train.h5')
        hdf_train = h5py.File(hdf_path,'r')
        self.x =hdf_train['features'][:]
        self.labels = [s.decode() for s in hdf_train['labels'][:]]
        self.nframe = self.x.shape[1]
        class_set = set(self.labels)
        self.y = class_to_int(self.labels,class_set)
        self.x,self.y,_ = balance_class_distribution_sub(self.x,self.y)

    def generate_eval(self):

        return self.feature_scale(self.x), self.y

class Datagen_test(Datagen):

    def __init__(self,hf,conf):
        super(Datagen_test, self).__init__(conf= conf, is_test=True)
        
        self.x_pos_1 = hf['feat_pos_0'][:]
        self.x_pos_2 = hf['feat_pos_1'][:]
        self.x_pos_3 = hf['feat_pos_2'][:]
        self.x_pos_4 = hf['feat_pos_3'][:]
        self.x_pos_5 = hf['feat_pos_4'][:]

        self.x_neg_1 = hf['feat_neg_0'][:]
        self.x_neg_2 = hf['feat_neg_1'][:]
        self.x_neg_3 = hf['feat_neg_2'][:]
        self.x_neg_4 = hf['feat_neg_3'][:]
        self.x_neg_5 = hf['feat_neg_4'][:]
        
        self.x_query = hf['feat_query'][:]
        
    def generate_eval(self):

        '''Returns normalizedtest features

        Output:
        - X_pos: Positive set features. Positive class prototypes will be calculated from this
        - X_query: Query set. Onset-offset prediction will be made on this set.
        - X_neg: The entire audio file. Will be used to calculate a negative prototype.
        '''
        X_pos_1 = (self.x_pos_1)
        X_pos_2 = (self.x_pos_2)
        X_pos_3 = (self.x_pos_3)
        X_pos_4 = (self.x_pos_4)
        X_pos_5 = (self.x_pos_5)
        X_neg_1 = (self.x_neg_1)
        X_neg_2 = (self.x_neg_2)
        X_neg_3 = (self.x_neg_3)
        X_neg_4 = (self.x_neg_4)
        X_neg_5 = (self.x_neg_5)
        X_query = (self.x_query)
        
        X_pos_1 = self.feature_scale(X_pos_1)
        X_pos_2 = self.feature_scale(X_pos_2)
        X_pos_3 = self.feature_scale(X_pos_3)
        X_pos_4 = self.feature_scale(X_pos_4)
        X_pos_5 = self.feature_scale(X_pos_5)
        X_neg_1 = self.feature_scale(X_neg_1)
        X_neg_2 = self.feature_scale(X_neg_2)
        X_neg_3 = self.feature_scale(X_neg_3)
        X_neg_4 = self.feature_scale(X_neg_4)
        X_neg_5 = self.feature_scale(X_neg_5)
        X_query = self.feature_scale(X_query)
      
        return X_pos_1,X_pos_2,X_pos_3,X_pos_4,X_pos_5,X_neg_1,X_neg_2,X_neg_3,X_neg_4,X_neg_5,X_query

class Datagen_KD(Datagen):

    def __init__(self,hf,conf):
        super(Datagen_KD, self).__init__(conf= conf)
        self.x = hf['x_train'][:]
        self.y = hf['y_train'][:]

    def generate_eval(self):

        '''Returns normalizedtest features

        Output:
        - X_pos: Positive set features. Positive class prototypes will be calculated from this
        - X_query: Query set. Onset-offset prediction will be made on this set.
        - X_neg: The entire audio file. Will be used to calculate a negative prototype.
        '''
        X = (self.x)
        Y = (self.y)
        X = self.feature_scale(X)
        return X,Y















