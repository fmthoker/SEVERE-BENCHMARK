# SEVERE Benchmark

Official code for our ECCV 2022 paper [How Severe is Benchmark-Sensitivity in Video
Self-Supervised Learning?](https://arxiv.org/abs/2203.14221)

TL;DR. We propose the SEVERE (<ins>SE</ins>nsitivity of <ins>V</ins>id<ins>E</ins>o <ins>RE</ins>presentations) benchmark for evaluating the generalizability of representations obtained by existing and future self-supervised video learning methods.

![](./media/concept_figure.png)


## Overview of Experiments 
We evaluate 9 video self-supervised learning (VSSL) methods  on  7 video datasets  for 6 video understanding tasks.

### Evaluated VSSL models

Below are the video self-suprevised methods  that we evaluate.

| Model | URL |
|-------|-----|
| SeLaVi| https://github.com/facebookresearch/selavi |
| MoCo| https://github.com/tinapan-pt/VideoMoCo |
| VideoMoCo | https://github.com/tinapan-pt/VideoMoCo |
| Pretext-Contrast | https://github.com/BestJuly/Pretext-Contrastive-Learning  |
| RSPNet | https://github.com/PeihaoChen/RSPNet |
| AVID-CMA | https://github.com/facebookresearch/AVID-CMA |
| CtP | https://github.com/microsoft/CtP |
| TCLR | https://github.com/DAVEISHAN/TCLR |
| GDT | https://github.com/facebookresearch/GDT |
| Supervised | https://pytorch.org/vision/0.8/_modules/torchvision/models/video/resnet.html#r2plus1d_18 |


* For SeLaVi, MoCo, VideoMoCO, Pretext-Contrast, CtP, TCLR and GDT we use the Kinetics-400 pretrained R(2+1D)-18 weights provided by the Authors.
* For RSPNet and AVID-CMA the author provided R(2+1D)-18 weights differ from the R(2+1D)-18 architecture defined in ['A Closer Look at Spatiotemporal Convolutions for Action Recognition'](https://arxiv.org/abs/1711.11248). Thus we use the official implementation of the RSPNet and AVID-CMA and to pretrain with the common R(2+1D)-18 backbone on Kinetics-400 dataset.
* For Supervised, We use the Kinetics-400 pretrained R(2+1D)-18 weights from the pytorch library.

Download Kinetics-400 pretrained R(2+1D)-18 weights for each method from [here](https://surfdrive.surf.nl/files/index.php/s/Zw9tbuOYAInzVQC). Unzip the downloaded file and it shall create a folder `checkpoints_pretraining/` with all weights.

## Experiments

We divide these downstream evaluations across four axes:

### I. Downstream domain-shift

We analyse whether features learned by self-supervised models transfer to datasets that vary in domain ([UCF101 ](http://crcv.ucf.edu/data/UCF101.php),[Something_something_v2](https://developer.qualcomm.com/software/ai-datasets/something-something),[NTU-60](https://rose1.ntu.edu.sg/dataset/actionRecognition/),[Fine-Gym v_1.0](https://sdolivia.github.io/FineGym/),etc) with respect to the pre-training dataset i.e. [Kinetics-400](https://arxiv.org/abs/1705.06950).

Please refer to [ssl_benchmark/README.md](./ssl_benchmark/README.md) for steps to reproduce the experiments with varying downstream domain.

### II. Downstream sample-sizes

We evaluate the sensitivity of self-supervised methods to the number of downstream samples available for finetuning.

Please refer to [ssl_benchmark/README.md](./ssl_benchmark/README.md) for steps to reproduce the experiments with varying downstream samples.

### III. Downstream action granularities

We investigate whether self-supervised methods can learn fine-grained features required for recognizing semantically similar actions.

Please refer to [ssl_benchmark/README.md](./ssl_benchmark/README.md) for steps to reproduce the experiments with varying downstream actions.

### IV. Downstream task-shift

We study the sensitivity of video self-supervised methods to the downstream task and question whether self-supervised features can be used beyond action recognition.

**In-domain tasks**: For task-shift on in-domain datasets, we use tasks such as repetition counting on UCF101 which has .  Please refer to [Deep-Temporal-Repetition-Counting/README.md](./Deep-Temporal-Repetition-Counting/README.md) for steps to reproduce experiments.

**Out-of-domain tasks**: We use multi-label classification on Charades and action detection on AVA as examples of task-shift on domains far away from the standard UCF101. Please refer to [SlowFast-ssl-vssl/README.md](./SlowFast-ssl-vssl/README.md) for steps to reproduce the experiments with tasks (a) action detection (AVA), (b) multi-label classification (Charades).

## The SEVERE Benchmark

From our analysis we distill the SEVERE-benchmark, a subset of our experiments, that can be useful for evaluating current and future video representations beyond standard benchmarks.


### Citation

If you use our work or code, kindly consider citing our paper:
```
@InProceedings{Thoker:2022:SEVERE:ECCV,
    author    = {Thoker, Fida and Doughty, Hazel and Bagad, Piyush and Snoek, Cees},
    title     = {How Severe is Benchmark-Sensitivity in Video Self-Supervised Learning?},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month     = {October},
    year      = {2022},
    pages     = {-}
}
```


### Acknowledgements

### Maintainers

* [Fida Thoker](https://fmthoker.github.io/)
* [Piyush Bagad](https://bpiyush.github.io/)

:bell: If you face an issue or have suggestions, please create a Github issue and we will try our best to address soon.
