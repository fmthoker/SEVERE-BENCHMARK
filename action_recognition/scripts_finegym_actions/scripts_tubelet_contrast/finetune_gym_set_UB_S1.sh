
python finetune.py configs/benchmark/gym_set_UB_S1/112x112x32.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/  --seed 100

python test.py configs/benchmark/gym_set_UB_S1/112x112x32_test.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ 
