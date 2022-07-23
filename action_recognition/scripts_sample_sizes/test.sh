#set test_only flag to true in the  config file 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml    --pretext-model-name  scratch --finetune-ckpt-path ./checkpoints/scratch/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml   --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./checkpoints/selavi/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml   --pretext-model-name  pretext_contrast --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt --finetune-ckpt-path ./checkpoints/pretext_contrast/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples_avid.yaml   --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./checkpoints/avid_cma/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml   --pretext-model-name  moco --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/moco/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml   --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml   --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml   --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar  --finetune-ckpt-path ./checkpoints/rspnet/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples_tclr.yaml   --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints/tclr/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml   --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./checkpoints/ctp/ 

python test.py configs/benchmark/gym99/112x112x32_1000_examples.yaml   --pretext-model-name  supervised --pretext-model-path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth --finetune-ckpt-path ./checkpoints/fully_supervised_kinetics/ 

