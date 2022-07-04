#  Expirements Action Recognition 
### Evaluating self-supervised video representation models for Domain Shift, Sample Sizes and Fine-grained action classification. 

## Requirements

* pip install -r requirements.txt 

## Pretrained Models
* Please download our pretrained models  [here](https://surfdrive.surf.nl/files/index.php/s/Zw9tbuOYAInzVQC).
```bash
mv checkpoints_pretraining/ ..
```

## Dataset Preparation
The datasets can be downloaded from the following links:

* [UCF101 ](http://crcv.ucf.edu/data/UCF101.php)
* [Something_something_v2](https://developer.qualcomm.com/software/ai-datasets/something-something)
* [NTU-60](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
* For [Fine-Gym v_1.0](https://sdolivia.github.io/FineGym/) please send a request to Fine-Gym authors via [Form](https://docs.google.com/forms/d/e/1FAIpQLScg8KDmBl0oKc7FbBedT0UJJHxpBHQmgsKpc4nWo4dwdVJi0A/viewform) to get access to the dataset. After downloading the videos, follow  the script provided in [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README.md) to trim the videos to subactions. (Note, if you also dowload the videos via [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README.md) script some of the video will be  missing because of the broken youtube links, so we suggest to request the Fine-Gym authors to get access to whole dataset.)

* We provide the annoations that we use for each dataset in the ./data/ directory:
* The expected directory hierarchy is as follow:
```
├── data
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
│   │       │   
│   │       └── ...
│   │──smth-smth-v2
│   │   ├── 20bn-something-something-v2
│   │   │   └── *.avi
│   │   └── something-something-v2-annotations
│   │       ├── something-something-v2-labels.json
│   │       ├── something-something-v2-test.json
│   │       ├── something-something-v2-train.json
│   │       └── something-something-v2-validation.json
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
└── ...
```



## Domain Shift 
### Training 
* For finetuning pretrained models on domain shift datasets (e.g something_something_v2,gym_99, etc) use training scripts in  ./scripts_domain_shift/
```bash
# Example finetuning pretrained  gdt model on something-something-v2 
python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ --seed 100
```
### Testing
```bash
# After finetuning, set test_only flag to true in the  config file (e.g configs/benchmark/something/112x112x32.yaml)  and run
python test.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/
```
## Sample size 
* For finetuning pretrained models with different sample sizes use training scripts in  ./scripts_sample_sizes
```bash

# Training
# Example finetuning pretrained  gdt model on 1000 ucf101 examples  
python finetune.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --seed 100
# Note, set flag 'num_of_examples: to N'in the corresponding config file (e.g configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml) to use N samples.

# Testing,  set test_only flag to true in the  config file and run
python test.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --seed 100
```

## Fine-gym Granularities 


### Acknowledgements
 We use parts of  code from : [Audio-Visual Instance Discrimination with Cross-Modal Agreement](https://github.com/facebookresearch/AVID-CMA) for buliding this repo. 
