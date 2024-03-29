from datasets import load_dataset

dataset = load_dataset("agkphysics/AudioSet", "unbalanced")
sample_audio_test = dataset['test'][0]['audio']
print(sample_audio_test['array'])