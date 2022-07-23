

## Setup

* Create conda environment and install dependencies:
    ```sh
    bash setup/create_env.sh slowfast-charades
    ```
    This will create and activate a conda environment called `slowfast-charades`.
    Please activate it for further steps.
    ```sh
    conda activate slowfast-charades
    ```
* Symlink the dataset folder. Suppose Charades and AVA datasets are stored inside `/path/to/datasets/`. Then, run the following from the repo:
    ```sh
    ln -s /path/to/datasets/ data
    ```
* Symlink the pre-trained models for initialization. Suppose all your VSSL pre-trained checkpoints are at `/path/to/checkpoints_pretraining`
    ```sh
    ls -s /path/to/checkpoints_pretraining/ checkpoints_pretraining
    ```


## Experiments on Charades

### Fine-tuning a pre-trained VSSL model

To run fine-tuning on Charades, using `r2plus1d_18` backbone initialized from Kinetics-400 supervised pretraining, we use the following command(s):
```sh
conda activate slowfast-charades
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