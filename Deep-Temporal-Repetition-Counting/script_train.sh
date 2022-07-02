
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_from_scratch  --pretext_model_name scratch

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_selavi  --pretext_model_name selavi  --pretext_model_path ../checkpoints_pretraining/selavi/selavi_kinetics.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_pretext_contrast  --pretext_model_name pretext_contrast --pretext_model_path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_avid_cma  --pretext_model_name avid_cma  --pretext_model_path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_moco  --pretext_model_name moco --pretext_model_path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_video_moco  --pretext_model_name video_moco --pretext_model_path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_ctp  --pretext_model_name ctp --pretext_model_path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_rspnet_snellius  --pretext_model_name rspnet --pretext_model_path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_tclr  --pretext_model_name tclr --pretext_model_path ../checkpoints_pretraining/tclr/rpd18kin400.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_gdt  --pretext_model_name gdt --pretext_model_path ../checkpoints_pretraining/gdt/gdt_K400.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --model resnet --model_depth 18 --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_full_supervision  --pretext_model_name supervised --pretext_model_path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth 
