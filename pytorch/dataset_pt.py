import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    #mode specify train, val or test
    #data is a dict and include all data used, use data['...'] to get specified types  
    def __init__(self,mode,*args, ** kwargs):
        super(BaseDataset, self).__init__()
        assert mode in [ "train", "val", "test"]
        self.mode = mode
        #This function is used to resolve data must be used
        loaded_data = self.load_data(self.mode, kwargs)
        self.loaded_data = loaded_data
        #for optional data, resolve here

        #for all data, please define transformation here

        

    #The function is used to resolve the data to get input data that can be directly used
    #return a dict
    def load_data(mode, **kwargs):
        pass
    
    @property
    def is_train(self):
        return self.mode == "train"

    #split the dataset to train, test, val(type Dataset)
    @classmethod
    def splits(cls,*arg, **kwargs):
        train = cls(mode="train", *arg, **kwargs)
        test = cls(mode="test", *arg, **kwargs)
        val = cls(mode="val",*arg, **kwargs)
        return train, test, val

    #Given an index, this function is used to give loaded_data[index]
    def __getitem__(self, index):
        #return loaded_data[index]
        pass
    
    #Return the number of the loaded_data
    def __len__(self):
        #return len(loaded_data)
        pass
    
    #@property
    #define property function here

    #define other functions here


