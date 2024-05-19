import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random as random
import musdb
import librosa
import glob
import os
import numpy as np
from data.dataset_musdb import dataset_musdb
from data.dataset_fma import dataset_fma


            
class dataset_musdb_sourcevae(Dataset):
    """musdb and fma dataset:
        for training, each sample contains:
                1. an unsupervised sample mixture
                2. contains supervised sample with probability P

    """
    def __init__(
        self,
        musdb_root='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18',
        sample_rate=16000,
        mode='train',
        seconds=4,
        source_types=['vocals', 'drums', 'bass', 'other'],
        len_ds=100
    ):
        # assert mode in ['train'], 'only train mode allowed'
        self.mode = mode
        self.ds_musdb = {}
        self.source_types = source_types
        for source_type in source_types:
            self.ds_musdb[source_type] = dataset_musdb(
                                            root_dir=musdb_root,
                                            sample_rate=sample_rate,
                                            mode=mode,
                                            mixture=False,
                                            source_types=[source_type],
                                            seconds=seconds,
                                            )
        self.len_ds = len_ds
    
    def __len__(self):
        return self.len_ds

    def __getitem__(self, idx):
        
        batch = {}
        
        current_idx = idx % len(self.ds_musdb)
        for source_type in self.source_types:
            if self.mode == 'train':
                current_idx = np.random.randint(low=0, high=self.len_ds)
            source_batch = self.ds_musdb[source_type][current_idx]
            batch[source_type] = source_batch[source_type]

        return batch
        

def collate_funcs_sourcevae(batches):
    '''
        collate function for musdb
    '''
    
    new_batch = {}
    for key in batches[0].keys():
        new_batch[key] = [] 
    for batch in batches:
        # print(key, len(new_batch[key]))
        for key in new_batch.keys():
            if key in batch.keys():
                new_batch[key].append(torch.tensor(batch[key], dtype=torch.float32))
    for key in new_batch.keys():
        new_batch[key] = torch.stack(new_batch[key], dim=0).unsqueeze(1)
    
    return new_batch



if __name__=='__main__':
    from tqdm import tqdm
    # dataset = dataset_vocalset(
    #     vocalset_root='/media/synrg/NVME-2TB/alanweiyang/datasets/vocalset11',
    #     sample_rate=16000,
    #     mode='train',
    #     seconds=4)

    # batch = dataset[30]
    # print(len(dataset))
    # sf.write('vocals.wav', batch['vocals'], 16000)
    # 148/148795.mp3
    # try:
    #     audio_path = '/data/romit/alan/fma/fma/data/fma_large/148/148795.mp3'
    #     audio, sr = librosa.load(audio_path, sr=16000, mono=True, offset=0, duration=10)
    # except Exception as e:
    #     print('stupid error', e)
        
    #     print('fuck it')
    # print(audio.shape, '----------------------------------------------------------------')
    
    dataset = dataset_musdb_sourcevae(
        musdb_root='/data/romit/alan/musdb18',
        sample_rate=16000,
        mode='train',
        seconds=4,
        source_types=['vocals', 'drums', 'bass', 'other'],
        len_ds=100
    )
    print(len(dataset))

    # bad_samples = []
    # for i in tqdm(range(200)):
    #     try:
    #         dataset[i]
    #     except Exception as e:
    #         bad_samples.append(i)
    #         print('bad sample: ', i)
    #         if i == len(dataset-1):
    #             break
    #         continue
    # print('bad samples', bad_samples)

    # batch = dataset[10000]    
    # for key in batch.keys():
    #     print(key, batch[key].shape)

    batch = collate_funcs_sourcevae([dataset[0], dataset[1], dataset[2]])
    print('batch')
    print(batch.keys())
    # batch['mixture]: bs, T
    # batchp['vocals]: bs, T
   

    for key in batch.keys():
        print(key, batch[key].shape)
    # print(len(dataset))
        # sf.write(key+'.wav', batch[key][0, 0], 16000)
    
    # batch = collate_func_vocals([dataset[30], dataset[31]])
    # print(batch['vocals'].shape)