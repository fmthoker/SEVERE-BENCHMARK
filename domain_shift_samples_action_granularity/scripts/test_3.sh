
python test.py configs/benchmark/gym_set_FX_S1/112x112x32_scratch.yaml  configs/main/scratch/pretext.yaml  --pretext-model-name  scratch 

python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml  configs/main/selavi/kinetics/pretext.yaml --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth

python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml  configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model-name  pretext_contrast --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt

python test.py configs/benchmark/gym_set_FX_S1/112x112x32_avid.yaml  configs/main/avid_cma/kinetics/pretext.yaml --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar

python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml  configs/main/moco/kinetics/pretext.yaml --pretext-model-name  moco --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar

python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml  configs/main/video_moco/kinetics/pretext.yaml --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar

python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth

python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml  configs/main/rspnet/kinetics/pretext.yaml --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar


python test.py configs/benchmark/gym_set_FX_S1/112x112x32_tclr.yaml  configs/main/tclr/kinetics/pretext.yaml --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth

python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml  configs/main/ctp/kinetics/pretext.yaml --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth

python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model-name  supervised --pretext-model-path ../checkpoints_pretraining/full_supervision/r2plus1d_18-91a641e6.pth
