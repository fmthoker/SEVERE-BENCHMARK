
python finetune.py configs/benchmark/gym99/112x112x32_avid.yaml  configs/main/avid_cma/kinetics/pretext.yaml --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --seed 100

python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/rspnet/kinetics/pretext.yaml --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --seed 100
