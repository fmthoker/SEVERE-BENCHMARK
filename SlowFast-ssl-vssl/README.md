
# Experiments on AVA and Charades

This sub-repo is based on Facebook's official [SlowFast repo](https://github.com/facebookresearch/SlowFast). We extend it to experiment with various video self-supervised learning methods to initialize weights of R(2+1)D backbone for AVA and Charades datasets. On AVA, the task is fine-grained action detection. On Charades, the task is multi-label action classification.

For more details about more broader usage of the SlowFast code, please refer to the [Getting started](https://github.com/facebookresearch/SlowFast/blob/main/GETTING_STARTED.md) guide in the official repo. All credits and copyrights are reserved with the original repository authors.

## Setup

* Create conda environment and install dependencies:
    ```sh
    bash setup/create_env.sh slowfast
    ```
    This will create and activate a conda environment called `slowfast`.
    Please activate it for further steps.
    ```sh
    conda activate slowfast
    ```
<!-- * (Refer to the following sections for setting up datasets) Symlink the dataset folder. Suppose Charades and AVA datasets are stored inside `/path/to/datasets/`. Then, run the following from the repo:
    ```sh
    ln -s /path/to/datasets/ data
    ``` -->
* Symlink the pre-trained models for initialization. Suppose all your VSSL pre-trained checkpoints are at `/path/to/checkpoints_pretraining`
    ```sh
    ls -s /path/to/checkpoints_pretraining/ checkpoints_pretraining
    ```


### Data preparation for AVA

Overall, the data preparation for AVA takes about 20 hours.

1. Symlink the data folder
```sh
cd /path/to/repo/
mkdir -p data

# example: /ssd/pbagad/datasets/AVA
export ROOT_DATA_DIR=/path/to/where/you/want/to/store/AVA-dataset

# symlink
ln -s $ROOT_DATA_DIR data/AVA/
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


## Experiments on Charades

### Fine-tuning a pre-trained VSSL model

To run fine-tuning on Charades, using `r2plus1d_18` backbone initialized from Kinetics-400 supervised pretraining, we use the following command(s):
```sh
conda activate slowfast
cd /path/to/repo/
export PYTHONPATH=$PWD

cfg=configs/Charades/VSSL/32x8_112x112_R18_supervised.yaml
bash scripts/jobs/train_on_charades.sh -c $cfg
```
This assumes that you have setup data folders symlinked into the repo. This shall save outputs in `./outputs/` folder. You can check `./outputs/<expt-folder-name>/logs/train_logs.txt` to see the training progress.

:warning: Note that, on Charades, we obtain all our results using 1 GPU (NVIDIA RTX A600, 48GBs each) and a batch size of 16.

:hourglass: Each experiment takes about 8 hours to run on the suggested configuration.


## Experiments on AVA

We run all our experiments on AVA 2.2.

### Fine-tuning a pre-trained VSSL model

To run fine-tuning on AVA, using `r2plus1d_18` backbone initialized from Kinetics-400 supervised pretraining, we use the following command(s):
```sh
bash scripts/jobs/train_on_ava.sh -c configs/AVA/VSSL/32x2_112x112_R18_v2.2_supervised.yaml
```

You can check out other configs for fine-tuning with other video self-supervised methods.

The training is followed by an evaluation on the test set. Thus, the numbers will be displayed in logs at the end of the run.

:warning: Note that, on AVA, we train using 4 GPUs (GeForce GTX 1080 Ti, 11GBs each) and a batch size of 32.

:hourglass: Each experiment takes about 8 hours to run on the suggested configuration.