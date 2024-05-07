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


            
class dataset_semi_supervised(Dataset):
    """musdb and fma dataset:
        for training, each sample contains:
                1. an unsupervised sample mixture
                2. contains supervised sample with probability P

    """
    def __init__(
        self,
        musdb_root='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18',
        fma_root='/data/romit/alan/fma/fma',
        sample_rate=16000,
        mode='train',
        seconds=4,
        p=1,
    ):
        # assert mode in ['train'], 'only train mode allowed'
        self.mode = mode
        self.ds_fma = dataset_fma(
            root_dir=fma_root,
            sample_rate=sample_rate,
            mode=mode,
            seconds=seconds,
            )
        self.ds_musdb = dataset_musdb(
            root_dir=musdb_root,
            sample_rate=sample_rate,
            mode=mode,
            mixture=True,
            source_types=['vocals', 'drums', 'bass', 'other'],
            seconds=seconds,
            )
        self.p = p
    
    def __len__(self):
        return len(self.ds_fma)

    def __getitem__(self, idx):
        mixture_fma = self.ds_fma[idx]
        
        batch = {}
        p_current = np.random.uniform()
        if p_current <= self.p:
            musdb_idx = idx % len(self.ds_musdb)
            batch = self.ds_musdb[musdb_idx]
        batch['mixture_fma'] = mixture_fma
        return batch
        

def collate_func_semi_supsevised(batches):
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
    
    dataset = dataset_semi_supervised(
        musdb_root='/data/romit/alan/musdb18',
        fma_root='/data/romit/alan/fma/fma',
        sample_rate=16000,
        mode='validation',
        seconds=10,
        p=1,
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

    batch = collate_func_semi_supsevised([dataset[0], dataset[1], dataset[2]])
    # batch['mixture]: bs, T
    # batchp['vocals]: bs, T
   

    for key in batch.keys():
        print(key, batch[key].shape)
    # print(len(dataset))
        sf.write(key+'.wav', batch[key][0, 0], 16000)
    
    # batch = collate_func_vocals([dataset[30], dataset[31]])
    # print(batch['vocals'].shape)