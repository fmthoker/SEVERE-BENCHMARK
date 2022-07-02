
python finetune.py  configs/benchmark/something/112x112x32_avid.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./checkpoints/avid_cma/ --seed 100


python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  moco --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/moco/ --seed 100


python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ --seed 100


python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ --seed 100

