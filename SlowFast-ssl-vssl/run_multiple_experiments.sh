# echo "Running experiment: TCLR"

# cfg_1=configs/Charades/VSSL/32x8_112x112_R18_tclr_seed_1.yaml
# cfg_2=configs/Charades/VSSL/32x8_112x112_R18_tclr_seed_2.yaml
# cfg_3=configs/Charades/VSSL/32x8_112x112_R18_tclr_seed_3.yaml

# CUDA_VISIBLE_DEVICES=0 bash scripts/jobs/train_on_charades.sh -c $cfg_1 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=1 bash scripts/jobs/train_on_charades.sh -c $cfg_2 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=2 bash scripts/jobs/train_on_charades.sh -c $cfg_3 -b 16 -n 1


# echo "Running experiment: MOCO"

# cfg_1=configs/Charades/VSSL/32x8_112x112_R18_moco_seed_1.yaml
# cfg_2=configs/Charades/VSSL/32x8_112x112_R18_moco_seed_2.yaml
# cfg_3=configs/Charades/VSSL/32x8_112x112_R18_moco_seed_3.yaml

# CUDA_VISIBLE_DEVICES=0 bash scripts/jobs/train_on_charades.sh -c $cfg_1 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=1 bash scripts/jobs/train_on_charades.sh -c $cfg_2 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=2 bash scripts/jobs/train_on_charades.sh -c $cfg_3 -b 16 -n 1

# echo "Running experiment: VIDEO_MOCO"

# cfg_1=configs/Charades/VSSL/32x8_112x112_R18_video_moco_seed_1.yaml
# cfg_2=configs/Charades/VSSL/32x8_112x112_R18_video_moco_seed_2.yaml
# cfg_3=configs/Charades/VSSL/32x8_112x112_R18_video_moco_seed_3.yaml

# CUDA_VISIBLE_DEVICES=0 bash scripts/jobs/train_on_charades.sh -c $cfg_1 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=1 bash scripts/jobs/train_on_charades.sh -c $cfg_2 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=2 bash scripts/jobs/train_on_charades.sh -c $cfg_3 -b 16 -n 1

# echo "Running experiment: PRETEXT_CONTRAST"

# cfg_1=configs/Charades/VSSL/32x8_112x112_R18_pretext_contrast_seed_1.yaml
# cfg_2=configs/Charades/VSSL/32x8_112x112_R18_pretext_contrast_seed_2.yaml
# cfg_3=configs/Charades/VSSL/32x8_112x112_R18_pretext_contrast_seed_3.yaml

# CUDA_VISIBLE_DEVICES=0 bash scripts/jobs/train_on_charades.sh -c $cfg_1 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=1 bash scripts/jobs/train_on_charades.sh -c $cfg_2 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=2 bash scripts/jobs/train_on_charades.sh -c $cfg_3 -b 16 -n 1

# echo "Running experiment: AVID_CMA"

# cfg_1=configs/Charades/VSSL/32x8_112x112_R18_avid_cma_seed_1.yaml
# cfg_2=configs/Charades/VSSL/32x8_112x112_R18_avid_cma_seed_2.yaml
# cfg_3=configs/Charades/VSSL/32x8_112x112_R18_avid_cma_seed_3.yaml

# CUDA_VISIBLE_DEVICES=0 bash scripts/jobs/train_on_charades.sh -c $cfg_1 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=1 bash scripts/jobs/train_on_charades.sh -c $cfg_2 -b 16 -n 1 &
# CUDA_VISIBLE_DEVICES=2 bash scripts/jobs/train_on_charades.sh -c $cfg_3 -b 16 -n 1

echo "Running experiment: SELAVI"

cfg_1=configs/Charades/VSSL/32x8_112x112_R18_selavi_seed_1.yaml
cfg_2=configs/Charades/VSSL/32x8_112x112_R18_selavi_seed_2.yaml
cfg_3=configs/Charades/VSSL/32x8_112x112_R18_selavi_seed_3.yaml

CUDA_VISIBLE_DEVICES=0 bash scripts/jobs/train_on_charades.sh -c $cfg_1 -b 16 -n 1 &
CUDA_VISIBLE_DEVICES=1 bash scripts/jobs/train_on_charades.sh -c $cfg_2 -b 16 -n 1 &
CUDA_VISIBLE_DEVICES=2 bash scripts/jobs/train_on_charades.sh -c $cfg_3 -b 16 -n 1
