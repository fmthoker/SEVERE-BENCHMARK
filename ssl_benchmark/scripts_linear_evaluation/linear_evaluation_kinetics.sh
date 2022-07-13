
python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./checkpoints/selavi/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  pretext_contrast --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt --finetune-ckpt-path ./checkpoints/pretext_contrast/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  moco --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/moco/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear_tclr.yaml   --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints/tclr/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./checkpoints/ctp/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  supervised --pretext-model-path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth --finetune-ckpt-path ./checkpoints/fully_supervised_kinetics/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./checkpoints/avid_cma/ 

python linear_eval.py configs/benchmark/kinetics/112x112x32-linear.yaml   --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ 

