import torch
from torch.utils.data import DataLoader

class BaseDataLoader(DataLoader):

    @classmethod
    def splits(cls, train_data, val_data, test_data, **kwargs):
        #for example
        # train_load = cls(
        #     dataset=train_data,                                                               # define the dataset used
        #     batch_size=batch_size * num_gpus,                                                 # batch_size: attention! "*num_gpus"
        #     shuffle=True,                                                                     # shuffle data or not
        #     num_workers=num_workers,                                                          # num_cpus or gpus used to loader
        #     collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),  # transform data (for example, split None)
        #     drop_last=True,                                                                   # if not enough for one batch, drop or not
        #     pin_memory=True,                                                                  # store data directly in memory without in disk
        #     **kwargs,                                                                           (need more memory, but faster)
        # )
        # val_load = cls(
        #     dataset=val_data,
        #     batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
        #     shuffle=False,
        #     num_workers=num_workers,
        #     collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
        #     drop_last=True,
        #     # pin_memory=True,
        #     **kwargs,
        # )
        # return train_load, val_load

    #define self collate_fn, return x after transformed
    def my_collate(self,x):
        pass