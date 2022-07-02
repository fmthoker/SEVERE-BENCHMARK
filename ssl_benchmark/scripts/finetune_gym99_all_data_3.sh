
python finetune.py configs/benchmark/gym99/112x112x32_tclr.yaml  configs/main/tclr/kinetics/pretext.yaml --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --seed 100

python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/ctp/kinetics/pretext.yaml --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --seed 100

python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model-name  supervised --pretext-model-path ../checkpoints_pretraining/full_supervision/r2plus1d_18-91a641e6.pth --seed 100
