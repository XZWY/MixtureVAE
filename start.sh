export PYTHONPATH=/ws/ifp-54_2/hasegawa/xulinf2/MixtureVAE

export CUDA_VISIBLE_DEVICES=1
musdb_root_path='/ws/ifp-54_2/hasegawa/xulinf2/datasets/musdb18'
vocalset_root_path='/ws/ifp-54_2/hasegawa/xulinf2/datasets/vocalset11'
config_path="/ws/ifp-54_2/hasegawa/xulinf2/MixtureVAE/config_SourceVAE.json"
# config_path="/ws/ifp-54_2/hasegawa/xulinf2/MixtureVAE/log_files/log_test_vae_kl_codec_100_256bn/config.json"
log_root="/ws/ifp-54_2/hasegawa/xulinf2/MixtureVAE/log_files/log_test_vae_kl_codec_1000_128bn"

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
