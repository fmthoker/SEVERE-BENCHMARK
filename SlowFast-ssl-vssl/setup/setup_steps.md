### Clone the repository

### Install the dependencies

### Data preparation for AVA

Overall, the data preparation for AVA takes about 20 hours.

1. Symlink the data folder
```sh
cd /path/to/repo/
mkdir -p data

# example: /ssd/pbagad/datasets/AVA
export ROOT_DATA_DIR=/path/to/where/you/want/to/store/AVA-dataset

# symlink
ln -s $ROOT_DATA_DIR data/ava/
```

2. Download: This step takes about 3.5 hours.
```sh
cd scripts/prepare-ava/
bash download_data.sh
```

3. Cut each video from its 15th to 30th minute: This step takes about 14 hours.
```sh
bash cut_videos.sh
```

4. Extract frames: This step takes about 1 hour.
```sh
bash extract_frames.sh
```

5. Download annotations: This step takes about 30 minutes.
```sh
bash download_annotations.sh
```

6. Setup exception videos that may have failed the first time. For me, there was this video `I8j6Xq2B5ys.mp4` that failed the first time. See `scripts/prepare-ava/exception.sh` to re-run the steps for such videos.

### Data preparation for Charades

This, overall, takes about 2 hours.

1. Symlink the data folder
```sh
ln -s /ssd/pbagad/datasets/charades data/charades
```

2. Download and unzip RGB frames
```sh
cd scripts/prepare-charades/
bash download_data.sh
```

3. Download the split files
```sh
bash download_annotations.sh
```

### Running experiments

Suppose you want to run an experiment with the following config on DAS-5:
```sh
cfg=configs/AVA/CTP/das5_32x2_112x112_R18_v2.2.yaml
```

Note that this config has checkpoint path on DAS-5 machine. In case you want to run the experiment on your machine, you need to change the path to your checkpoint in the config. Ideally, you should create a new config with this change.
```yaml
MODEL:
  CKPT: /path/to/your/checkpoint
```

We run training and evaluation by a job script such as `scripts/jobs/das5_train_on_ava.sh`. For a new machine,
create a new file and change the following variables in the script. Don't forget to copy data to local SSD. 
```sh
# configure output directory
output_dir=/var/scratch/pbagad/expts/SlowFast-ssl/$expt_folder

# configure data related arguments
FRAME_DIR="/local-ssd/pbagad/datasets/AVA/frames/"
FRAME_LIST_DIR="/local-ssd/pbagad/datasets/AVA/annotations/"
ANNOTATION_DIR="/local-ssd/pbagad/datasets/AVA/annotations/"
```

Setup and run training (example shown for DAS-5):
```sh
cd /path/to/repo/

# On DAS5
conda activate slowfast-gpu

# Run training
bash scripts/jobs/das5_train_on_ava.sh -c configs/AVA/CTP/das5_32x2_112x112_R18_v2.2.yaml -b 32

# Run evaluation after training is complete
bash scripts/jobs/das5_test_on_ava.sh -c configs/AVA/CTP/das5_32x2_112x112_R18_v2.2.yaml
```

* Changing batch size: In case the default batch size does not fit GPU memory, you can change it by passing `-b BATCH_SIZE` to the above command. For 32 frames, I have used batch size as 32.
* Changing number of GPUs can be done using `-n` argument.

#### Generic training script

To run training on any machine, we also provide a simple job script that can be used.
```sh
bash scripts/jobs/train_on_ava.sh -c CONFIG_PATH -b BATCH_SIZE -d DATA_DIR -o OUTPUT_DIR
```
For example, to run training on DAS-5, you can use the following command:
```sh
bash scripts/jobs/train_on_ava.sh -c configs/AVA/CTP/das5_32x2_112x112_R18_v2.2.yaml -b 32 -d /local-ssd/pbagad/datasets/AVA/ -o /var/scratch/pbagad/expts/SlowFast-ssl/
```

### Running fine-tuning on Charades

To run fine-tuning on Charades, using `r2plus1d_18` backbone initialized from Kinetics-400 supervised pretraining, we use the following command:
```sh
cfg=configs/Charades/R2PLUS1D/32x8_112x112_R18.yaml
bash scripts/jobs/train_on_charades.sh -c $cfg
```
This assumes that you have setup data folders symlinked into the repo. This shall save outputs in `./outputs/` folder.