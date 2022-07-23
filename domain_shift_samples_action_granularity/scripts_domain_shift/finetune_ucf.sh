
python finetune.py configs/benchmark/ucf/112x112x32-fold1_scratch.yaml    --pretext-model-name  scratch --finetune-ckpt-path ./checkpoints/scratch/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1.yaml   --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./checkpoints/selavi/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1.yaml   --pretext-model-name  pretext_contrast --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt --finetune-ckpt-path ./checkpoints/pretext_contrast/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1.yaml   --pretext-model-name  moco --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/moco/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1.yaml   --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1.yaml   --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1_tclr.yaml   --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints/tclr/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1.yaml   --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./checkpoints/ctp/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1.yaml   --pretext-model-name  supervised --pretext-model-path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth --finetune-ckpt-path ./checkpoints/fully_supervised_kinetics/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1_avid.yaml   --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./checkpoints/avid_cma/ --seed 100

python finetune.py configs/benchmark/ucf/112x112x32-fold1.yaml   --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ --seed 100

