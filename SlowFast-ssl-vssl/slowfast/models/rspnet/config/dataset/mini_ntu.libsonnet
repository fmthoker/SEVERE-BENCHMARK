local normalization = import "normalization.libsonnet";

{
    name: 'mini_ntu120',
    root: '/ssdstore/fmthoker/ntu/ntu120_transcoded',
    annotation_path: '/ssdstore/fmthoker/ntu/mini_ntu_120_cross_view_TrainTestlist',
    fold: 1,
    num_classes: 120,

    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}
