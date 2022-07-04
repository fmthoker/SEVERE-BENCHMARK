local ntu60 = import '../dataset/mini_ntu.libsonnet';
local r2plus1d = import '../model/r2plus1d.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: ntu60,
    model: r2plus1d,
    model_type: 'multitask',
    temporal_transforms+: {
        size: 16,
        frame_rate: null
    },
    local batch_size_factor =112*112*8 / self.temporal_transforms.size / self.spatial_transforms.size / self.spatial_transforms.size,
    batch_size: 16 * batch_size_factor,
    validate: {
        batch_size: 128 * batch_size_factor,
    },
    final_validate: {
        batch_size: 16 * batch_size_factor,
    },
    optimizer+: {lr: 0.01, schedule: "multi_step", milestones: [100,150]},
    num_epochs: 200,
}
