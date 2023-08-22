
# R(2+1)D-18

# Mini Kinetics
#python finetune.py configs/benchmark/diving48/112x112x32.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/mini-kinetics/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ --seed 12345
#
#python test.py configs/benchmark/diving48/112x112x32_test.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/mini-kinetics/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/



# Kinetics 400
python finetune.py configs/benchmark/diving48/112x112x32.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ --seed 12345

python test.py configs/benchmark/diving48/112x112x32_test.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/




