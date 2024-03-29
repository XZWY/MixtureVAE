from datasets import load_dataset

dataset = load_dataset("agkphysics/AudioSet", "unbalanced")

for i in range(100):
    sample_audio_test = dataset['train'][i]
    print(sample_audio_test['human_labels'])
# {
#  'video_id': '--PJHxphWEs',
#  'audio': {
#   'path': 'audio/bal_train/--PJHxphWEs.flac',
#   'array': array([-0.04364824, -0.05268681, -0.0568949 , ...,  0.11446512,
#           0.14912748,  0.13409865]),
#   'sampling_rate': 48000
#  },
#  'labels': ['/m/09x0r', '/t/dd00088'],
#  'human_labels': ['Speech', 'Gush']
# }

# print(sample_audio_test['array'].shape)
# print(sample_audio_test.keys())
# # dict_keys(['path', 'array', 'sampling_rate'])
# print(sample_audio_test['path'])
# print(sample_audio_test['array'].shape)
# print(sample_audio_test['sampling_rate'])