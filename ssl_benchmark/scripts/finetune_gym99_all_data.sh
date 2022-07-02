

python finetune.py configs/benchmark/gym99/112x112x32_scratch.yaml  configs/main/scratch/pretext.yaml  --pretext-model-name  scratch --seed 100

python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/selavi/kinetics/pretext.yaml --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --seed 100

python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model-name  pretext_contrast --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt --seed 100




