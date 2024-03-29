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

class dataset_vocalset(Dataset):
    """vocalset dataset:
    """

    def __init__(
        self,
        vocalset_root='/media/synrg/NVME-2TB/alanweiyang/datasets/vocalset11',
        sample_rate=16000,
        mode='train',
        seconds=4
    ):
        self.seconds = seconds
        self.sample_rate = sample_rate
        self.mode = mode
        
        
        # train, validation, test splits accoding to vocalset
        if self.mode=='train':
            self.subjects = ['female1','female3','female4','female5','female6','female7','female9','male1','male2','male4','male6','male7','male8','male9','male11']
        elif self.mode=='validation':
            # self.subjects = ['female7','female9','male1','male2']
            self.subjects = ['female2','female8']
        elif self.mode=='test':
            self.subjects = ['male3','male5','male10']
        else:
            assert False, 'train, validation, and test'
        
        self.vocalset_filelists = []
        for subject in self.subjects:
            self.vocalset_filelists += glob.glob(os.path.join(vocalset_root, 'FULL', subject, '*', '*', '*.wav'))

    def __len__(self):
        return len(self.vocalset_filelists)

    def __getitem__(self, idx):
        current_file = self.vocalset_filelists[idx]
        
        audio, sr = librosa.load(current_file, sr=self.sample_rate)
        
        total_samples = int(self.seconds * sr)
        # If the audio is longer than T seconds, crop it randomly
        
        batch = {}
        if len(audio) > total_samples:
            # Maximum start index for cropping
            max_start_idx = len(audio) - total_samples
            # Randomly choose a start index
            start_idx = np.random.randint(0, max_start_idx)
            # Crop the audio
            cropped_audio = audio[start_idx:start_idx + total_samples]
            batch['vocals'] = cropped_audio
        # If the audio is shorter than T seconds, pad it randomly
        else:
            # Calculate the number of samples to pad
            pad_samples = total_samples - len(audio)
            # Randomly choose padding at the beginning or the end
            pad_before = np.random.randint(0, pad_samples + 1)
            pad_after = pad_samples - pad_before
            # Pad the audio
            padded_audio = np.pad(audio, (pad_before, pad_after), 'constant', constant_values=(0, 0))
            batch['vocals'] = padded_audio
        return batch
            
class dataset_vocal(Dataset):
    """vocalset and musdb dataset:
        only allows train and validation mode
        for each sample in this dataset, there are one sample from vocalset and one sample from musdb
    """
    def __init__(
        self,
        musdb_root='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18',
        vocalset_root='/media/synrg/NVME-2TB/alanweiyang/datasets/vocalset11',
        sample_rate=16000,
        mode='train',
        seconds=4
    ):
        assert mode in ['train', 'validation'], 'only train and validation modes allowed'
        self.mode = mode
        self.ds_vocalset = dataset_vocalset(
            vocalset_root=vocalset_root,
            sample_rate=sample_rate,
            mode=mode,
            seconds=seconds
            )
        self.ds_musdb = dataset_musdb(
            root_dir=musdb_root,
            sample_rate=sample_rate,
            mode=mode,
            mixture=False,
            source_types=['vocals'],
            seconds=seconds,
            )
    
    def __len__(self):
        return len(self.ds_vocalset)

    def __getitem__(self, idx):
        batch_vocalset = self.ds_vocalset[idx]
        
        if self.mode=='train':
            musdb_idx = np.random.randint(0, len(self.ds_musdb))
        elif self.mode=='validation':
            musdb_idx = idx % len(self.ds_musdb)
        musdb_audio = self.ds_musdb[musdb_idx]['vocals']
        vocalset_audio = self.ds_vocalset[idx]['vocals']
        batch = {}
        batch['vocals'] = np.stack([vocalset_audio, musdb_audio])
        return batch
        


def collate_func_vocals(batches):
    '''
        collate function for musdb
    '''
    new_batch = {}
    for key in batches[0].keys():
        new_batch[key] = []
    for batch in batches:
        # print(key, len(new_batch[key]))
        for key in new_batch.keys():
            new_batch[key].append(torch.tensor(batch[key], dtype=torch.float32))
    for key in new_batch.keys():
        new_batch[key] = torch.cat(new_batch[key], dim=0).unsqueeze(1)
        
    return new_batch


if __name__=='__main__':
    # dataset = dataset_vocalset(
    #     vocalset_root='/media/synrg/NVME-2TB/alanweiyang/datasets/vocalset11',
    #     sample_rate=16000,
    #     mode='train',
    #     seconds=4)

    # batch = dataset[30]
    # print(len(dataset))
    # sf.write('vocals.wav', batch['vocals'], 16000)
    
    dataset = dataset_vocal(
        musdb_root='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18',
        vocalset_root='/media/synrg/NVME-2TB/alanweiyang/datasets/vocalset11',
        sample_rate=16000,
        mode='train',
        seconds=4,
    )
    print(len(dataset))
    
    batch = dataset[30]
    print(len(dataset))
    sf.write('vocals1.wav', batch['vocals'][0], 16000)
    sf.write('vocals2.wav', batch['vocals'][1], 16000)
    
    batch = collate_func_vocals([dataset[30], dataset[31]])
    print(batch['vocals'].shape)