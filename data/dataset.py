import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import librosa
import glob

# 1.A for separation: FUSS/ssdata/train_example_list.txt
# 4. musdb18/train/*.mp4, musdb18/test/*.mp4
class dataset_single_source(Dataset):
    """Dataset contains sources from datasets below:    
        1. FUSS/fsd_data/train/sound/*wav, FUSS/fsd_data/validation/sound/*wav,  FUSS/fsd_data/eval/sound/*wav
        2. FSD2018/FSDKaggle2018.audio_train/*.wav, FSD2018/FSDKaggle2018.audio_test/*.wav
        3. read_speech/*.wav
    """

    def __init__(
        self,
        dataset_dir,
        sample_rate=16000,
        datasets=['fuss', 'fsd', 'readspeech'],
        mode='train',
        segments=4
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.datasets = datasets
        self.mode = mode
        self.segments = segments
        
        self.file_lists = {}
        
        for dataset in datasets:
            if dataset == 'fuss':
                self.file_lists['fuss'] = glob(os.path.join(dataset_dir, 'FUSS/fsd_data', mode, 'sound', '*.wav'))
            elif dataset == 'fsd':
                self.file_lists['fsd'] = glob(os.path.join(dataset_dir, 'FSD2018/FSDKaggle2018.audio_' + mode, '*.wav'))
            elif dataset == 'readspeech':
                self.file_lists['readspeech'] = glob(os.path.join(dataset_dir, 'read_speech', '*.wav'))
            else:
                raise ValueError("Unknown dataset provided: {}".format(dataset))



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None and self.mode=='train':
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # If task is enh_both then the source is the clean mixture

        # Read sources
        if self.mode == 'train':
            n_src = np.random.randint(self.min_num_sources, self.max_num_sources+1)
        else:
            n_src = int(idx % 3 + 1)
        shuffle_indices = (np.arange(0, 3))
        np.random.shuffle(shuffle_indices)
        for i in range(n_src):
            source_path = row[f"source_{shuffle_indices[i] + 1}_path"]
            s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)
        
        # load noise if noisy
        if self.noisy:
            noise_path = self.df.iloc[idx]["noise_path"]
            noise, _ = sf.read(noise_path, dtype="float32", start=start, stop=stop)
        # # Read the mixture
        # mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # # Convert to torch tensor
        # mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        
        tgt_idx = np.random.randint(0, n_src)
        target = sources[tgt_idx]
        
        # create noisy mixture        
        mixture = sources.sum(0)
        if self.noisy:
            mixture = mixture + noise
        
        # get subband feature
        subband_target_snr = np.random.uniform(self.subband_snr_range[0], self.subband_snr_range[1])
        noise_interference = mixture - target
        target_sb = librosa.resample(librosa.resample(target, 16000, self.subband_size*2), self.subband_size*2, 16000)
        noise_interference_sb = librosa.resample(librosa.resample(noise_interference, 16000, self.subband_size*2), self.subband_size*2, 16000)
        alpha = np.linalg.norm(target_sb) / (np.linalg.norm(noise_interference_sb) * 10**(subband_target_snr / 20))
        alpha = np.clip(alpha, 0, 0.9)
        noisy_target_sb = target_sb + alpha * noise_interference_sb

        min_len = np.min([noisy_target_sb.shape[0], target.shape[0]])
        target = target[:min_len]
        noisy_target_sb = noisy_target_sb[:min_len]
        target_sb = target_sb[:min_len]

        
        # print(alpha, subband_target_snr)
        snr = 20 * np.log10(np.linalg.norm(target_sb) / (np.linalg.norm(noisy_target_sb-target_sb) + 1e-8))
        
        batch = {}
        batch['snr'] = torch.tensor(snr).type(torch.float32)
        # Convert sources to tensor
        mixture = torch.from_numpy(mixture)
        target = torch.from_numpy(target)
        target_sb = torch.from_numpy(target_sb)
        target_sb_noisy = torch.from_numpy(noisy_target_sb)
        
        batch["n_src"] = torch.tensor(n_src).type(torch.float32)
        batch['mixture'] = mixture
        batch['target'] = target
        batch['target_sb'] = target_sb
        batch['target_sb_noisy'] = target_sb_noisy
        
        return batch

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self._dataset_name()
        infos["task"] = self.task
        return infos

    def _dataset_name(self):
        """Differentiate between 2 and 3 sources."""
        return f"Libri{self.n_src}Mix"


def collate_func_separation(batches):
    '''
        collate function for general speech separation and enhancement
    '''
    new_batch = {}
    for key in batches[0].keys():
        new_batch[key] = [] 
    for batch in batches:
        # print(key, len(new_batch[key]))
        for key in new_batch.keys():
            new_batch[key].append(batch[key])
    for key in new_batch.keys():
        new_batch[key] = torch.stack(new_batch[key], dim=0)
    
    return new_batch
    
if __name__=='__main__':
    
    # * ``'enh_single'`` for single speaker speech enhancement.
    # * ``'enh_both'`` for multi speaker speech enhancement.
    # * ``'sep_clean'`` for two-speaker clean source separation.
    # * ``'sep_noisy'`` for two-speaker noisy source separation.
    
    # dataset = LibriMix(csv_dir='/workspace/host/LibriMix/dataset/Libri2Mix/wav16k/min/metadata', sample_rate=16000, n_src=1, segment=5, return_id=False, mode='train')
    # batch = collate_func_separation([dataset[0]])
    # # print(batch)
    # for key in batch.keys():
    #     if type(batch[key]) is torch.Tensor:
    #         print(key, batch[key].shape)
    
    # batch = dataset[0]
    # for key in batch.keys():
    #     if type(batch[key]) is torch.Tensor:
    #         print(key, batch[key].shape)
            
    # sf.write('mixture.wav', batch['mixture'], 16000)
    # sf.write('source0.wav', batch['sources'][0], 16000)
    # sf.write('source1.wav', batch['sources'][1], 16000)
    
    dataset = LibriMixIMUV(
        csv_dir='/workspace/host/LibriMix/dataset/Libri3Mix/wav16k/min/metadata',
        sample_rate=16000,
        noisy=True,
        min_num_sources=1,
        max_num_sources=3,
        subband_snr_range=[3, 10],
        # segment=5,
        mode='dev',
        subband_size=500
    )
    # max_values = []
    # for i in range(50):
    #     batch = dataset[i]
    #     max_value = batch['mixture'].max()
    #     print(max_value)
    #     max_values.append(max_value)
    # print('mean of max', torch.tensor(max_values).mean())
    
    batch = collate_func_separation([dataset[0]])
    batch = dataset[0]
    # print(batch)
    print(len(dataset))
    print(batch['snr'])
    print(batch['n_src'])
    for key in batch.keys():
        if type(batch[key]) is torch.Tensor:
            print(key, batch[key].shape)
    sf.write('mixture.wav', batch['mixture'], 16000)
    sf.write('target.wav', batch['target'], 16000)
    sf.write('target_sb.wav', batch['target_sb'], 16000)
    sf.write('target_sb_noisy.wav', batch['target_sb_noisy'], 16000)