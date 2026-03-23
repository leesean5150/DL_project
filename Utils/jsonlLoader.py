import torch
from torch.utils.data.dataloader import DataLoader
import jsonlines

class TransactionDataLoader(DataLoader):
    def __init__(self, dataset, batch_size = 1, shuffle = None, sampler = None, batch_sampler = None, num_workers = 0, collate_fn = None, pin_memory = False, drop_last = False, timeout = 0, worker_init_fn = None, multiprocessing_context=None, generator=None, *, prefetch_factor = None, persistent_workers = False, pin_memory_device = "", in_order = True):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device, in_order=in_order)

    def __get__(self, instance, owner):
        pass
    
    def __len__(self):
        pass