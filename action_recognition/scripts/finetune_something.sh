
#python finetune.py configs/benchmark/something/112x112x32_scratch.yaml  --pretext-model-name  scratch --finetune-ckpt-path ./checkpoints/scratch/ --seed 100

#python finetune.py configs/benchmark/something/112x112x32.yaml  --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./checkpoints/selavi/ --seed 100

#python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  pretext_contrast --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt --finetune-ckpt-path ./checkpoints/pretext_contrast/ --seed 100

python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ --seed 100




