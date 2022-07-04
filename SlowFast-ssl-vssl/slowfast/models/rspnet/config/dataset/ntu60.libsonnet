local normalization = import "normalization.libsonnet";

{
    name: 'ntu60',
    root: '/ssdstore/fmthoker/ntu/ntu120_transcoded',
    annotation_path: '/ssdstore/fmthoker/ntu/ntu_60_cross_view_TrainTestlist',
    fold: 1,
    num_classes: 60,

    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}
