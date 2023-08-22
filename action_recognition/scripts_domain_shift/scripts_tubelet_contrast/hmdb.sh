
# R(2+1)D-18

#python finetune.py configs/benchmark/hmdb51/112x112x32-fold1.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ --seed 12345

#python test.py configs/benchmark/hmdb51/112x112x32-fold1_test.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/

# R3D-18

#python finetune.py configs/benchmark/hmdb51/112x112x32-fold1.yaml   --pretext-model-name  tubelet_contrast_r3d --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r3d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ --backbone r3d_18 --seed 12345
#
#python test.py configs/benchmark/hmdb51/112x112x32-fold1_test.yaml   --pretext-model-name  tubelet_contrast_r3d --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r3d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ --backbone r3d_18 



# I3D

python finetune_i3d.py configs/benchmark/hmdb51/112x112x32-fold1_i3d.yaml   --pretext-model-name  motion_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/i3d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ --seed 12345 --dropout 0.5 

python test_i3d.py configs/benchmark/hmdb51/112x112x32-fold1_i3d_test.yaml   --pretext-model-name  motion_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/i3d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ --dropout 0.5 
