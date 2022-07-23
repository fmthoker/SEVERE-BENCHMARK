
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_full_supervision  --pretext_model_name supervised --pretext_model_path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth 
