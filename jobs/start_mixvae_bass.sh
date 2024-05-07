export PYTHONPATH=/data/romit/alan/MixtureVAE

export CUDA_VISIBLE_DEVICES=2,3
musdb_root_path='/data/romit/alan/musdb18'
vocalset_root_path='/data/romit/alan/vocalset11'
config_path="/data/romit/alan/MixtureVAE/configs/config_MixtureVAE_bass.json"
# config_path="/ws/ifp-54_2/hasegawa/xulinf2/MixtureVAE/log_files/log_test_vae_kl_codec_100_256bn/config.json"
log_root="/data/romit/alan/MixtureVAE/log_files/log_mixvae_bass"


# export PYTHONPATH=/media/synrg/NVME-2TB/alanweiyang/MixVAE

# export CUDA_VISIBLE_DEVICES=1,2
# musdb_root_path='/media/synrg/NVME-2TB/alanweiyang/datasets/musdb18'
# vocalset_root_path='/media/synrg/NVME-2TB/alanweiyang/datasets/vocalset11'
# config_path="/media/synrg/NVME-2TB/alanweiyang/MixVAE/config_SourceVAE_sb2.json"
# # config_path="/ws/ifp-54_2/hasegawa/xulinf2/MixtureVAE/log_files/log_test_vae_kl_codec_100_256bn/config.json"
# log_root="/media/synrg/NVME-2TB/alanweiyang/MixVAE/log_files/log_subband2"


echo "Train model...train_mixturevae.....single case, bass"
python3 ../train_mixture_vae_single.py \
    --musdb_root ${musdb_root_path}\
    --vocalset_root ${vocalset_root_path}\
    --config ${config_path} \
    --checkpoint_path ${log_root} \
    --checkpoint_interval 5000 \
    --summary_interval 5 \
    --validation_interval 1500 \
    --training_epochs 5000 \
    --vanilla_steps 50000 \
    --stdout_interval 5 \
