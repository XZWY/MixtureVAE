export PYTHONPATH=/media/synrg/NVME-2TB/alanweiyang/MixVAE

export CUDA_VISIBLE_DEVICES=0,1
musdb_root_path='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18'
vocalset_root_path='/media/synrg/NVME-2TB/alanweiyang/datasets/vocalset11'
config_path="/media/synrg/NVME-2TB/alanweiyang/MixVAE/config_SourceVAE.json"
log_root="/media/synrg/NVME-2TB/alanweiyang/MixVAE/log_files/log_test_vae_kl_codec_100"

echo "Train model..."
python train_source_vae_kl_codec.py \
    --musdb_root ${musdb_root_path}\
    --vocalset_root ${vocalset_root_path}\
    --config ${config_path} \
    --checkpoint_path ${log_root} \
    --checkpoint_interval 5000 \
    --summary_interval 5 \
    --validation_interval 5000 \
    --training_epochs 5000 \
    --vanilla_steps 50000 \
    --stdout_interval 5 \
