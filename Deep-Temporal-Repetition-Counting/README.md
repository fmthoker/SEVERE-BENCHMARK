# IV. Downstream Task-shift (in-domain)

Evaluating self-supervised video representation models for the task of Deep-Temporal-Repetition-Counting.

This sub-repo is based on official implementation of ["Context-aware and Scale-insensitive Temporal Repetition Counting"](https://github.com/Xiaodomgdomg/Deep-Temporal-Repetition-Counting). We extend it to experiment with various video self-supervised learning methods to initialize weights of R(2+1)D-18 backbone.

## Table of Contents

* [Setup](#setup)
* [Evaluated VSSL models](#evaluated-vssl-models)
* [Dataset Preparation](#dataset-preparation)
    * [UCFRep](#ucfrep)
* [Training](#training)
* [Testing](#testing)
* [Acknowledgements](#acknowledgements)

## Setup

* Create a `conda` environment
* Install dependencies
```sh
pip install -r requirements.txt 
```

## Evaluated VSSL models

* To evaluate video self-supervised pre-training methods used in the paper, you need the pre-trained checkpoints for each method. We assume that these models are downloaded as instructed in the [main README](../README.md).

## Dataset Preparation

### UCFRep
* Please download the UCF101 dataset [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert UCF101 videos from avi to png files, put the png files to data/ori_data/ucf526/imgs/train and `data/ori_data/ucf526/imgs/val`

## Training
For finetuning pretrained models for the task of Repetition-Counting run as following:

```bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_from_scratch  --pretext_model_name scratch

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_selavi  --pretext_model_name selavi  --pretext_model_path ../checkpoints_pretraining/selavi/selavi_kinetics.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_pretext_contrast  --pretext_model_name pretext_contrast --pretext_model_path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_avid_cma  --pretext_model_name avid_cma  --pretext_model_path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_moco  --pretext_model_name moco --pretext_model_path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_video_moco  --pretext_model_name video_moco --pretext_model_path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_ctp  --pretext_model_name ctp --pretext_model_path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_rspnet_snellius  --pretext_model_name rspnet --pretext_model_path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_tclr  --pretext_model_name tclr --pretext_model_path ../checkpoints_pretraining/tclr/rpd18kin400.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_gdt  --pretext_model_name gdt --pretext_model_path ../checkpoints_pretraining/gdt/gdt_K400.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_full_supervision  --pretext_model_name supervised --pretext_model_path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth 

```

## Testing

python main.py --no_train --resume_path = path to the finetuned_checkpoint with best validation accuracy (check validations logs )

### Acknowledgement
 Our code is based on the official implementation of CVPR2020 paper: ["Context-aware and Scale-insensitive Temporal Repetition Counting"](https://github.com/Xiaodomgdomg/Deep-Temporal-Repetition-Counting). If you use this code please cite their work too as following:

```bibtex
 @InProceedings{Zhang_2020_CVPR,
    author = {Zhang, Huaidong and Xu, Xuemiao and Han, Guoqiang and He, Shengfeng},
    title = {Context-Aware and Scale-Insensitive Temporal Repetition Counting},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
} 

