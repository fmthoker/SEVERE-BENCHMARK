# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torch.distributed as dist

import utils.logger
from utils import main_utils
import yaml
import os
import numpy as np

def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(np.prod(p.size())))
    return desc

def freeze_backbone(model):

    for n, p in model.named_parameters():
        if 'fc' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

    return model


def prepare_environment(args, cfg, fold):
    if args.distributed:
        while True:
            try:
                dist.init_process_group(backend='nccl', init_method='tcp://localhost:{}'.format(args.port), world_size=args.world_size, rank=args.gpu)
                break
            except RuntimeError:
                args.port = str(int(args.port) + 1)

    eval_dir = '{}/eval-{}/fold-{:02d}'.format(args.finetune_ckpt_path,  cfg['benchmark']['name'], fold)
    os.makedirs(eval_dir, exist_ok=True)
    yaml.safe_dump(cfg, open('{}/config.yaml'.format(eval_dir), 'w'))

    logger = utils.logger.Logger(quiet=args.quiet, log_fn='{}/eval.log'.format(eval_dir), rank=args.gpu)
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))
    logger.add_line("=" * 30 + "   Config   " + "=" * 30)
    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  '+ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))
    print_dict(cfg)

    return eval_dir, logger


def build_dataloader(db_cfg, split_cfg, fold, num_workers, distributed):
    import torch.utils.data as data
    from datasets import preprocessing
    if db_cfg['transform'] == 'msc+color':
        video_transform = preprocessing.VideoPrep_MSC_CJ(
            crop=(db_cfg['crop_size'], db_cfg['crop_size']),
            num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
            pad_missing=True,
            augment=split_cfg['use_augmentation'],
            min_area=db_cfg['min_area'],
            color=db_cfg['color'],
            switch_channels = db_cfg['switch_channels'],
            normalize= db_cfg['normalize'],
        )
    elif db_cfg['transform'] == 'crop+color':
        video_transform = preprocessing.VideoPrep_Crop_CJ(
            crop=(db_cfg['crop_size'], db_cfg['crop_size']),
            num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
            pad_missing=True,
            augment=split_cfg['use_augmentation'],
            switch_channels = db_cfg['switch_channels'],
            normalize= db_cfg['normalize'],
        )
    else:
        raise ValueError

    import datasets
    if db_cfg['name'] == 'ucf101':
        dataset = datasets.UCF
    elif db_cfg['name'] == 'hmdb51':
        dataset = datasets.HMDB
    elif db_cfg['name'] == 'kinetics':
        dataset = datasets.Kinetics
    elif db_cfg['name'] == 'something':
        dataset = datasets.SOMETHING
    elif db_cfg['name'] == 'ntu60':
        dataset = datasets.NTU
    elif db_cfg['name'] == 'gym99':
        dataset = datasets.GYM99
    elif db_cfg['name'] == 'gym288':
        dataset = datasets.GYM288
    elif db_cfg['name'] == 'gym_event_vault':
        dataset = datasets.GYM_event_vault
    elif db_cfg['name'] == 'gym_event_floor_exercise':
        dataset = datasets.GYM_event_floor_exercise
    elif db_cfg['name'] == 'gym_set_fx_s1':
        dataset = datasets.GYM_set_FX_S1
    elif db_cfg['name'] == 'gym_set_ub_s1':
        dataset = datasets.GYM_set_UB_S1
    else:
        raise ValueError('Unknown dataset')

    db = dataset(
        subset=split_cfg['split'].format(fold=fold),
        return_video=True,
        video_clip_duration=db_cfg['clip_duration'],
        video_fps=db_cfg['video_fps'],
        video_transform=video_transform,
        return_audio=False,
        return_labels=True,
        mode=split_cfg['mode'],
        clips_per_video=split_cfg['clips_per_video'],
        num_of_examples = db_cfg['num_of_examples'], 
    )

    print(distributed)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    else:
        sampler = None

    drop_last = split_cfg['drop_last'] if 'drop_last' in split_cfg else True
    loader = data.DataLoader(
        db,
        batch_size=db_cfg['batch_size']  if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size']//split_cfg['clips_per_video']),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(sampler is None) and split_cfg['use_shuffle'],
        sampler=sampler,
        drop_last=drop_last
    )
    return loader


def build_dataloaders(cfg, fold, num_workers, distributed, logger):
    logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
    train_loader = build_dataloader(cfg, cfg['train'], fold, num_workers, distributed)
    logger.add_line(str(train_loader.dataset))

    logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
    test_loader = build_dataloader(cfg, cfg['test'], fold, num_workers, distributed)
    logger.add_line(str(test_loader.dataset))

    logger.add_line("=" * 30 + "   Dense DB   " + "=" * 30)
    dense_loader = build_dataloader(cfg, cfg['test_dense'], fold, num_workers, distributed)
    logger.add_line(str(dense_loader.dataset))

    return train_loader, test_loader, dense_loader


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.
        self.rank = rank

    def save(self, model, optimizer, scheduler, epoch, eval_metric=0.):
        if self.rank is not None and self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        main_utils.save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best=is_best, model_dir=self.checkpoint_dir)

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer, scheduler, restore_last=False, restore_best=False):
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        model.load_state_dict(ckp['state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
        #for i in range(start_epoch):
            #scheduler.step(epoch=None)
        return start_epoch

class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        outs = []
        for i in range(0, x.shape[0], self.batch_size):
            outs += [self.model(x[i:i + self.batch_size])]
        return torch.cat(outs, 0)
