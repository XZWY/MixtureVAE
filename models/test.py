from models.models_hificodec import *
from models.env import *
import json
from data.dataset_musdb import dataset_musdb, collate_func_musdb
from data.dataset_vocal import dataset_vocal, collate_func_vocals
from models.SourceVAESB import SourceVAESB
from models.SourceVAESubband2 import SourceVAE
from models.loss import loss_vae_reconstruction
from torch.utils.data import DistributedSampler, DataLoader

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

with open('../config_SourceVAE.json') as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

musdb_root = '/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18'
vocalset_root = '/media/synrg/NVME-2TB/alanweiyang/datasets/vocalset11'

if h.source_type=='vocals':
    dataset = dataset_vocal(
        musdb_root=musdb_root,
        vocalset_root=vocalset_root,
        sample_rate=16000,
        mode='train',
        seconds=4,
    )
else:
    dataset = dataset_musdb(
        root_dir=musdb_root,
        sample_rate=16000,
        mode='train',
        source_types=[h.source_type],
        mixture=False,
        seconds=4,
        len_ds=5000)
print(len(dataset))

collate_func = collate_func_vocals if h.source_type=='vocals' else collate_func_musdb
batch = collate_func([dataset[0], dataset[1]])

# input = torch.randn(2, 1, 16000*4)
# emb = encoder(input)
# output = generator(emb)

# print(get_n_params(encoder)/1e6)
# print(get_n_params(generator)/1e6)

# print(emb.shape, output.shape)

sourcevae = SourceVAE(h)
batch = sourcevae(batch)

train_loader = DataLoader(
            dataset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_func)

import soundfile as sf
for i, batch in enumerate(train_loader):
    if i > 1:
        break
    print(batch[h.source_type].shape)
    batch = sourcevae(batch)
    for key in batch.keys():
        print(key, batch[key].shape)

    # print(batch)
# from models.loss import loss_vae_reconstruction
# LOSS = loss_vae_reconstruction(h)
# y = torch.randn(2,1,16000)
# y_hat = torch.randn(2,1,16000)
# print(LOSS(y, y_hat).shape)
