
python finetune.py configs/benchmark/gym99/112x112x32_1000_examples_tubelet.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/  --seed 100

python test.py configs/benchmark/gym99/112x112x32_1000_examples_tubelet_test.yaml   --pretext-model-name  tubelet_contrast --pretext-model-path ../checkpoints_pretraining/tubelet_contrast/kinetics-400/r2+1d/epoch_100.pth --finetune-ckpt-path ./checkpoints/tubelet_contrast/ 
