
#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/moco/kinetics/pretext.yaml --pretext-model-name  moco --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar --seed 100

python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/video_moco/kinetics/pretext.yaml --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --seed 100

python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --seed 100

