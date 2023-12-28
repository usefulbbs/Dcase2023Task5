import torch
import numpy as np
import torch.utils.data as data

''' Episodic batch sampler adoted from https://github.com/jakesnell/prototypical-networks/'''

class EpisodicBatchSampler(data.Sampler):
    def __init__(self, labels,masks, n_episodes, n_way, n_samples):
        '''
        Sampler that yields batches per n_episodes without replacement.
        Batch format: (c_i_1, c_j_1, ..., c_n_way_1, c_i_2, c_j_2, ... , c_n_way_2, ..., c_n_way_n_samples)
        Args:
            label: List of sample labels (in dataloader loading order)
            n_episodes: Number of episodes or equivalently batch size
            n_way: Number of classes to sample
            n_samples: Number of samples per episode (Usually n_query + n_support)
        '''
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_samples = n_samples

        labels = np.array(labels)
        self.samples_indices = {}
        for i in range(len(labels)):
            mask = masks[i]
            label = np.where(mask, labels[i], 0)
            label = list(set(label))
            if 0 in label: label.remove(0)
            if -1 in label: label.remove(-1)
            assert len(label)<=1
            if len(label)==1:
                cls = label[0]
                if cls not in self.samples_indices.keys():
                    self.samples_indices[cls]=[i]
                else:
                    self.samples_indices[cls].append(i)
        self.class_set = list(self.samples_indices.keys())
        self.n_way = n_way
    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for batch in range(self.n_episodes):
            batch = []
            classes = [self.class_set[i] for i in \
                    torch.randperm(len(self.samples_indices.keys()))[:self.n_way].tolist()] # torch.randperm(n) returns an array from 0 to n-1
            for c in classes:
                l = torch.tensor(self.samples_indices[c])
                pos = torch.randperm(len(l))[:self.n_samples]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1) # c*n_samples
            yield batch