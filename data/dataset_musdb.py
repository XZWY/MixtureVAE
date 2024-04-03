import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random as random
import musdb
import librosa

class dataset_musdb(Dataset):
    """musdb18 dataset:
        train: 100 songs
        testing: 50
    """

    def __init__(
        self,
        root_dir='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18',
        sample_rate=16000,
        mode='train',
        mixture=True,
        source_types=['vocals', 'drums', 'bass', 'other'],
        seconds=4,
        len_ds=100,
    ):
        self.len_ds = len_ds
        self.seconds = seconds
        self.sample_rate = sample_rate
        self.source_types = source_types
        self.mixture = mixture
        self.mode = mode
        
        if mode=='train':    
            self.mus_tracks = musdb.DB(subsets="train", split="train", root=root_dir).tracks
        elif mode=='validation':
            self.mus_tracks = musdb.DB(subsets="train", split="valid", root=root_dir).tracks
        elif self.mode=='test':
            self.mus_tracks = musdb.DB(subsets="test", root=root_dir).tracks
        else:
            assert False, 'train, validation, and test'
        self.signal_power_threshold={
            'vocals':0.015**2,
            'drums':0.001**2,
            'bass':0.001**2,
            'other':0.001**2,
        }

    def __len__(self):
        return self.len_ds

    def __getitem__(self, idx):
        idx = idx % len(self.mus_tracks) # so that the dataset has more samples
        track = self.mus_tracks[idx]
        # print(track, track.duration)
        if self.mode=='train':
            track.chunk_duration = self.seconds
            track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        elif self.mode=='validation':
            track.chunk_duration = self.seconds
            track.chunk_start = 0
        elif self.mode=='test':
            track.chunk_duration = track.duration
            track.chunk_start = 0
        
        while True:
            valid=True
            # setup starting point
            if self.mode=='train':
                track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
            elif self.mode=='validation':
                track.chunk_start = track.chunk_start + self.seconds
                assert track.chunk_start + self.seconds < track.duration, 'sample contains no sound'                
                    
            
            batch = {}
            for source_type in self.source_types:
                current_audio = librosa.resample(track.targets[source_type].audio.T[0,:], orig_sr=track.rate, target_sr=self.sample_rate)
                signal_power = (current_audio**2).mean()
                if signal_power < self.signal_power_threshold[source_type]: # the whole segment is silent
                    valid=False
                    # print(idx, source_type, 'fail this time, start changing')
                    # sf.write('sounds/failures/'+source_type+str(idx)+'.wav', current_audio, 16000)
                    break
                batch[source_type] = current_audio
            if not valid:
                continue
            if self.mixture:
                batch["mixture"] = librosa.resample(track.audio.T[0,:], orig_sr=track.rate, target_sr=self.sample_rate)
            break
        return batch
        
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
    dataset = dataset_musdb(
        root_dir='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18',
        sample_rate=16000,
        mode='train',
        source_types=['other'],
        mixture=False,
        seconds=4,
        len_ds=100
        )
    batch = dataset[0]
    sf.write('other'+'.wav', batch['other'], 16000)
    # c = {}
    # source_types=['vocals', 'drums', 'bass', 'other']
    # for source_type in source_types:
    #     c[source_type] = []
    # for i in tqdm(range(len(dataset))):
    #     for j in range(2):
    # # for i in tqdm(range(2)):
    # #     for j in range(2):
    #         batch = dataset[i]
    #         for source_type in source_types:
    #             c[source_type].append(batch[source_type].max())
    # for source_type in source_types:
    #     # print(source_type, c[source_type])
    #     plt.figure()
    #     plt.hist(c[source_type])
    #     plt.savefig(source_type+'.png')