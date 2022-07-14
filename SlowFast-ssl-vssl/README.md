
# IV. Downstream Task-shift

This sub-repo is based on Facebook's official [SlowFast repo](https://github.com/facebookresearch/SlowFast). We extend it to experiment with various video self-supervised learning methods to initialize weights of R(2+1)D backbone for AVA and Charades datasets. On AVA, the task is fine-grained action detection. On Charades, the task is multi-label action classification.

For more details about more broader usage of the SlowFast code, please refer to the [Getting started](https://github.com/facebookresearch/SlowFast/blob/main/GETTING_STARTED.md) guide in the official repo. All credits and copyrights are reserved with the original repository authors.

## Setup

We use `conda` to manage dependencies. If you have not installed `anaconda3` or `miniconda3`, please install it before following the steps below.

* Clone the repository (if not already done)
    ```sh
    git clone https://github.com/fmthoker/SEVERE_BENCHMARK.git
    ```
    Go to the experiment folder for AVA and Charades.
    ```sh
    cd SlowFast-ssl-vssl/
    ```
* Create conda environment and install dependencies:
    ```sh
    bash setup/create_env.sh slowfast
    ```
    This will create and activate a conda environment called `slowfast`.
    
    :warning: We use `torch==1.9.0` with CUDA 11.1. If you have a CUDA with a different version, kindly use an apt [PyTorch version](https://pytorch.org/get-started/previous-versions/). You will need to follow the steps in `setup/create_env.sh` manually to install different versions of dependencies.
    
    Please activate the environment for further steps.
    ```sh
    conda activate slowfast
    ```
<!-- * (Refer to the following sections for setting up datasets) Symlink the dataset folder. Suppose Charades and AVA datasets are stored inside `/path/to/datasets/`. Then, run the following from the repo:
    ```sh
    ln -s /path/to/datasets/ data
    ``` -->

## Evaluated VSSL models

To evaluate video self-supervised pre-training methods used in the paper, you need the pre-trained checkpoints for each method. We assume that these models are downloaded as instructed in the [main README](../README.md).

Download those from [here](https://surfdrive.surf.nl/files/index.php/s/Zw9tbuOYAInzVQC), if not already downloaded. Unzip the downloaded file. This will create `checkpoints_pretraining/` folder which contains checkpoints for each of the methods used in the paper.

* Symlink the pre-trained models for initialization. Suppose all your VSSL pre-trained checkpoints are at `../checkpoints_pretraining`
    ```sh
    ls -s ../checkpoints_pretraining/ checkpoints_pretraining
    ```

## Dataset preparation

The data processing steps for each of AVA and Charades datasets is quite tedious. The steps for each of them are listed below.

First, you need to create a symlink to the root dataset folder into the repo. For e.g., if you store all your datasets at `/path/to/datasets/`, then,
```sh
# make sure you are inside the `SlowFast-ssl-vssl/` folder in the repo
ln -s /path/to/datasets/ data
```

### [AVA](https://research.google.com/ava/download.html)

:hourglass: Overall, the data preparation for AVA takes about 20 hours.

These steps are based on the ones in original repo.

1. Download: This step takes about 3.5 hours.
```sh
cd scripts/prepare-ava/
bash download_data.sh
```

2. Cut each video from its 15th to 30th minute: This step takes about 14 hours.
```sh
bash cut_videos.sh
```

3. Extract frames: This step takes about 1 hour.
```sh
bash extract_frames.sh
```

4. Download annotations: This step takes about 30 minutes.
```sh
bash download_annotations.sh
```

5. Setup exception videos that may have failed the first time. For me, there was this video `I8j6Xq2B5ys.mp4` that failed the first time. See `scripts/prepare-ava/exception.sh` to re-run the steps for such videos.

### [Charades](https://prior.allenai.org/projects/charades)

:hourglass: This, overall, takes about 2 hours.

1. Download and unzip RGB frames
```sh
cd scripts/prepare-charades/
bash download_data.sh
```

2. Download the split files
```sh
bash download_annotations.sh
```


## Experiments on AVA

We run all our experiments on AVA 2.2.

### Fine-tuning a pre-trained VSSL model

To run fine-tuning on AVA, using `r2plus1d_18` backbone initialized from Kinetics-400 supervised pretraining, we use the following command(s):
```sh
cfg=configs/AVA/VSSL/32x2_112x112_R18_v2.2_supervised.yaml
bash scripts/jobs/train_on_ava.sh -c $cfg
```

You can check out other configs for fine-tuning with other video self-supervised methods. The configs for all pre-training methods is provided below:

| **Model**        | **Config**                                    |
|------------------|-----------------------------------------------|
| No pre-training  | `32x2_112x112_R18_v2.2_scratch.yaml`          |
| SeLaVi           | `32x2_112x112_R18_v2.2_selavi.yaml`           |
| MoCo             | `32x2_112x112_R18_v2.2_moco.yaml`             |
| VideoMoCo        | `32x2_112x112_R18_v2.2_video_moco.yaml`       |
| Pretext-Contrast | `32x2_112x112_R18_v2.2_pretext_contrast.yaml` |
| RSPNet           | `32x2_112x112_R18_v2.2_rspnet.yaml`           |
| AVID-CMA         | `32x2_112x112_R18_v2.2_avid_cma.yaml`         |
| CtP              | `32x2_112x112_R18_v2.2_ctp.yaml`              |
| TCLR             | `32x2_112x112_R18_v2.2_tclr.yaml`             |
| GDT              | `32x2_112x112_R18_v2.2_gdt.yaml`              |
| Supervised       | `32x2_112x112_R18_v2.2_supervised.yaml`       |

The training is followed by an evaluation on the test set. Thus, the numbers will be displayed in logs at the end of the run.

:warning: Note that, on AVA, we train using 4 GPUs (GeForce GTX 1080 Ti, 11GBs each) and a batch size of 32.

:hourglass: Each experiment takes about 8 hours to run on the suggested configuration.


## Experiments on Charades

### Fine-tuning a pre-trained VSSL model

To run fine-tuning on Charades, using `r2plus1d_18` backbone initialized from Kinetics-400 supervised pretraining, we use the following command(s):
```sh
# activate the environment
conda activate slowfast

# make sure you are inside the `SlowFast-ssl-vssl/` folder in the repo
export PYTHONPATH=$PWD

cfg=configs/Charades/VSSL/32x8_112x112_R18_supervised.yaml
bash scripts/jobs/train_on_charades.sh -c $cfg
```
This assumes that you have setup data folders symlinked into the repo. This shall save outputs in `./outputs/` folder. You can check `./outputs/<expt-folder-name>/logs/train_logs.txt` to see the training progress.

:warning: Note that, on Charades, we obtain all our results using 1 GPU (NVIDIA RTX A600, 48GBs each) and a batch size of 16.

:hourglass: Each experiment takes about 8 hours to run on the suggested configuration.

