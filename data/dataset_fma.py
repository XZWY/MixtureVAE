import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random as random
import librosa

import numpy as np
import pandas as pd
import data.data_utils as utils
import os

class dataset_fma(Dataset):
    """fma dataset:
         dataloader for the fma_large dataset
    """

    def __init__(
        self,
        root_dir='/data/romit/alan/fma/fma',
        sample_rate=16000,
        mode='train',
        seconds=4,
    ):
        trach_csv_path = os.path.join(root_dir, 'data/fma_metadata/tracks.csv')
        # genres_csv_path = os.path.join(root_dir, 'data/fma_metadata/genres.csv')
        # features_csv_path = os.path.join(root_dir, 'data/fma_metadata/features.csv')
        # echonest_csv_path = os.path.join(root_dir, 'data/fma_metadata/echonest.csv')

        tracks = utils.load(trach_csv_path)
        if mode=='train':
            self.tracks = tracks[tracks['set','split']=='training']
        elif mode=='validation':
            self.tracks = tracks[tracks['set','split']=='validation']
        else:
            self.tracks = tracks[tracks['set','split']=='test']
        # genres = utils.load(genres_csv_path)
        # features = utils.load(features_csv_path)
        # echonest = utils.load(echonest_csv_path)

        self.audio_dir = os.path.join(root_dir, 'data/fma_large')

        self.sample_rate = sample_rate
        self.seconds = seconds
        

    def __len__(self):
        return self.tracks.shape[0]

    def __getitem__(self, idx):
        id = self.tracks.iloc[idx].name
        current_file = utils.get_audio_path(self.audio_dir, id)

        total_seconds = 29.9 # each segment has 30 seconds inside
        max_start_second = total_seconds - self.seconds
        start_second = np.random.randint(0, max_start_second)

        try:
            audio, sr = librosa.load(current_file, sr=self.sample_rate, mono=True, offset=start_second, duration=self.seconds)
        except Exception as e:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        if audio.shape[-1] != int(self.seconds * self.sample_rate):
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        return audio
        # total_samples = int(self.seconds * sr)
        # # If the audio is longer than T seconds, crop it randomly
        
        # batch = {}
        # if len(audio) > total_samples:
        #     # Maximum start index for cropping
        #     max_start_idx = len(audio) - total_samples
        #     # Randomly choose a start index
        #     start_idx = np.random.randint(0, max_start_idx)
        #     # Crop the audio
        #     cropped_audio = audio[start_idx:start_idx + total_samples]
        #     batch['vocals'] = cropped_audio
        # # If the audio is shorter than T seconds, pad it randomly
        # else:
        
        
def collate_func_musdb(batches):
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
        new_batch[key] = torch.stack(new_batch[key], dim=0).unsqueeze(1)
    
    return new_batch


if __name__=='__main__':
    # source_type = 'bass'
    # dataset = dataset_musdb(
    #     root_dir='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18',
    #     sample_rate=16000,
    #     mode='train',
    #     source_types=[source_type],
    #     mixture=False,
    #     seconds=4)
    # dataset[9]
    # for i in range(len(dataset)):
    #     batch = dataset[i]
    #     sf.write('sounds/'+source_type+str(i)+'.wav', batch[source_type], 16000)
    # batch = dataset[0]
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    dataset = dataset_fma(
        root_dir='/data/romit/alan/fma/fma',
        sample_rate=16000,
        mode='train',
        seconds=4,
        )

    for i in tqdm(range(len(dataset))):
        print(i, dataset[i].shape)

    # bad_samples = []
    # for i in tqdm(range(len(dataset))):
    #     try:
    #         dataset[i]
    #     except Exception as e:
    #         bad_samples.append(i)
    #         print('bad sample: ', i)
    #         # if i == 199:
    #         #     break
    #         continue
    # print('bad samples', bad_samples)


    # print(len(dataset))
    # audio = dataset[9000]
    # print('audio', audio.shape)
    # sf.write('fma.wav', audio, 16000)
    