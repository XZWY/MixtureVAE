export PYTHONPATH=/data/romit/alan/MixtureVAE

export CUDA_VISIBLE_DEVICES=6,7
musdb_root_path='/data/romit/alan/musdb18'
vocalset_root_path='/data/romit/alan/vocalset11'
config_path="/data/romit/alan/MixtureVAE/config_SourceVAE.json"
# config_path="/ws/ifp-54_2/hasegawa/xulinf2/MixtureVAE/log_files/log_test_vae_kl_codec_100_256bn/config.json"
log_root="/data/romit/alan/MixtureVAE/log_files/log_test_vae_kl_codec_10_128bn"

echo "Train model..."
python3 train_source_vae_kl_codec.py \
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
