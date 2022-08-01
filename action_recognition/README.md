<!---
We evaluate on various subsets defined for  [Fine-Gym](https://sdolivia.github.io/FineGym/) dataset.
# Experiments on Action Recognition
-->
# I.   Downstream Domain-shift  <br />II.  Downstream Sample Sizes <br />III. Downstream Action granularities 
Evaluating self-supervised video representation models for the task of action recognition with varying  downstream domains, size of training set and fine-grained action labels. 

## Table of Contents

* [Setup](#setup)
* [Evaluated VSSL models](#evaluated-vssl-models)
* [Dataset Preparation](#dataset-preparation)
* [Experiments](#experiments)
    * [I. Downstream domain-shift](#i-downstream-domain-shift)
    * [II. Downstream sample-sizes](#ii-downstream-sample-sizes)
    * [III. Downstream fine-grained action classification](#iii-downstream-fine-grained-action-classification)
    * [Linear evaluation](#linear-evaluation)
* [Acknowledgements](#acknowledgements)

## Setup

We recommend creating a `conda` environment and installing dependencies in it by using:
```bash
conda create -n severe_env1 python=3.7
conda activate severe_env1
pip install -r requirements.txt 

```

We run our experiments on Python 3.7 and PyTorch 1.6. Other versions should work but are not tested.

## Evaluated VSSL models

* To evaluate video self-supervised pre-training methods used in the paper, you need the pre-trained checkpoints for each method. We assume that these models are at  path `../checkpoints_pretraining/` as instructed in the [main README](../README.md).

## Dataset Preparation

The datasets can be downloaded from the following links:

* [UCF101 ](http://crcv.ucf.edu/data/UCF101.php)
* [Something_something_v2](https://developer.qualcomm.com/software/ai-datasets/something-something)
* [NTU-60](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
* For [Fine-Gym v_1.0](https://sdolivia.github.io/FineGym/) please send a request to Fine-Gym authors via [Form](https://docs.google.com/forms/d/e/1FAIpQLScg8KDmBl0oKc7FbBedT0UJJHxpBHQmgsKpc4nWo4dwdVJi0A/viewform) to get access to the dataset. After downloading the videos, follow  the script provided in [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README.md) to trim the videos to subactions. (Note, if you also dowload the videos via [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README.md) script some of the video will be  missing because of the broken youtube links, so we suggest to request the Fine-Gym authors to get access to whole dataset). Please contact us in case of any issues or if you need preprocessed the Fine-Gym videos. 

* We provide the annoations that we use for each dataset in the ./data/ directory.
<!---
The expected directory hierarchy is as follow:-->
* We expect a directory hierarchy as below. After downloading the datasets from the original sources, please update the data and annotation paths for each dataset in the respective dataloader scripts e.g datasets/ucf.py, datasets/something.py, datasets/gym_99.py, etc. 
```
├── datasets_root
│   ├──ucf101
│   │   ├── ucfTrainTestlist
│   │   │   ├── classInd.txt
│   │   │   ├── testlist01.txt
│   │   │   ├── trainlist01.txt
│   │   │   └── ...
│   │   └── UCF-101
│   │       ├── ApplyEyeMakeup
│   │       │   └── *.avi
│   │       └── ...
│   ├──gym
│   │   ├── annotations
│   │   │   ├── gym99_train.txt
│   │   │   ├── gym99_val.txt 
│   │   │   ├── gym288_train.txt
│   │   │   ├── gym288_val.txt
│   │   │   └──
│   │   └── videos
│   │       ├── *.avi
│   │       └── ...
│   │
│   │──smth-smth-v2
│   │   ├── something-something-v2-annotations
│   │   │   ├── something-something-v2-labels.json
│   │   │   ├── something-something-v2-test.json
│   │   │   ├── something-something-v2-train.json
│   │   │   └── something-something-v2-validation.json
│   │   │       
│   │   └── something-something-v2-videos_avi
│   │       └── *.avi
│   │          
│   ├──ntu60
│   │   ├── ntu_60_cross_subject_TrainTestlist
│   │   │   ├── classInd.txt
│   │   │   ├── testlist01.txt
│   │   │   ├── trainlist01.txt
│   │   │   └── ...
│   │   └── videos
│   │       ├── brushing_hair
│   │       │   └── *.avi
│   │       ├── brushing_teeth
│   │       │   └── *.avi
│   │       └── ...
│   │
│   ├──kinetics-400
│   │   ├── labels
│   │   │   ├── train_videofolder.txt
│   │   │   ├── val_videofolder.txt
│   │   │   └── ...
│   │   └── VideoData
│   │       ├── playing_cards
│   │       │   └── *.avi
│   │       ├── singing
│   │       │   └── *.avi
│   │       └── ...
└── ...
```

## Experiments

Below, we run experiments for domain-shift, sample-sizes and fine-grained action classification.

### I. Downstream domain-shift

* For finetuning pretrained models on domain shift datasets (e.g `something_something_v2`, `gym_99`, etc) use training scripts in  [./scripts_domain_shift/](./scripts_domain_shift/).
```bash
# Example finetuning pretrained  gdt model on something-something-v2 

## Training 
python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ --seed 100
## Testing
# After finetuning, set test_only flag to true in the  config file (e.g configs/benchmark/something/112x112x32.yaml)  and run
python test.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/
```

### II. Downstream sample-sizes

* For finetuning pretrained models with different sample sizes use training scripts in  [./scripts_sample_sizes](./scripts_sample_sizes).

```bash
# Example finetuning pretrained  video_moco model with 1000 ucf101 examples  

# Training
python finetune.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ --seed 100
# Note, set flag 'num_of_examples: to N'in the corresponding config file (e.g configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml) if you want to change the number of training samples to N.

# Testing
#set test_only flag to true in the  config file and run
python test.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ 
```

### III. Downstream action-granularities

For this, we use the FineGym dataset that comes with a hierarchicy of actions.

* For finetuning pretrained models with different Fine-gym granularities (e.g `gym_event_vault`, `gym_set_FX_S1`, `gym288`, etc) use training scripts in  [./scripts_finegym_actions](./scripts_finegym_actions).

```bash
# Example finetuning pretrained  fully_supervised_kinetics model on set FX_S1  granularity

# Training
python finetune.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --seed 100

# Testing
#set test_only flag to true in the  config file and run
python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml   --pretext-model-name  supervised --pretext-model-path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth 
```

### Linear Evaluation 

* For evaluating pretrained models using linear evaluation on UCF-101 or Kinetics-400  use training scripts in  [./scripts_linear_evaluation](./scripts_linear_evaluation).

```bash
# Example linear evaluation on UCF-101

# Training
python linear_eval.py configs/benchmark/ucf/112x112x32-fold1-linear.yaml   --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ 

# Testing
#set test_only flag to true in the  config file and run
python test.py configs/benchmark/ucf/112x112x32-fold1-linear.yaml   --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ 
```

### Acknowledgements

 We use parts of  code from : [Audio-Visual Instance Discrimination with Cross-Modal Agreement](https://github.com/facebookresearch/AVID-CMA) for buliding this repo. 
