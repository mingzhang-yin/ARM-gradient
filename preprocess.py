import numpy as np
import os
import sys


class Dataset:

    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self.num_examples = data.shape[0]
        pass
    
    
    @property
    def data(self):
        return self._data
    
    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self.num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
    
        # go to the next batch
        if start + batch_size > self.num_examples:
            self._epochs_completed += 1
            rest_num_examples = self.num_examples - start
            data_rest_part = self.data[start:self.num_examples]
            idx0 = np.arange(0, self.num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
    
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]
      
def preprocess(dataset):
    return(Dataset(dataset))





