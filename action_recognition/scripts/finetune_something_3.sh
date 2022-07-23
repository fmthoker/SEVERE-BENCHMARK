
#python finetune.py  configs/benchmark/something/112x112x32_tclr.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints/tclr/ --seed 100

#python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./checkpoints/ctp/ --seed 100


python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  supervised --pretext-model-path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth  --finetune-ckpt-path ./checkpoints/full_supervision/ --seed 100


