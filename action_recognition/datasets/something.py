# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from datasets.video_db import VideoDataset
import os
from pathlib import Path
from typing import Dict
import json
import numpy as np
import math

DATA_PATH = '/local-ssd/fmthoker/smth-smth-v2/something-something-v2-videos_avi'
ANNO_PATH = '/local-ssd/fmthoker/smth-smth-v2/something-something-v2-annotations/'



def read_class_idx(annotation_dir: Path) -> Dict[str, str]:
    class_ind_path = annotation_dir+'/something-something-v2-labels.json'
    with open(class_ind_path) as f:
        class_dict = json.load(f)
    return class_dict

from datasets.utils import get_subset_data

class SOMETHING(VideoDataset):
    def __init__(self, subset,
                 video_clip_duration=0.5,
                 return_video=True,
                 video_fps=16.,
                 video_transform=None,
                 return_audio=False,
                 return_labels=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=20,
                 num_of_examples=0,
                 ):

        assert return_audio is False
        self.name = 'SOMETHING'
        self.root = DATA_PATH
        self.subset = subset
        self.class_idx_dict = read_class_idx(ANNO_PATH)

        filenames = []
        labels = []
        if 'train' in subset:
               video_list_path = f'{ANNO_PATH}/something-something-v2-train.json' 
               with open(video_list_path) as f:
                   video_infos = json.load(f)
                   for video_info in video_infos:
                       video = int(video_info['id'])
                       video_name = f'{video}.avi'
                       class_name = video_info['template'].replace('[', '').replace(']', '')
                       class_index = int(self.class_idx_dict[class_name])
                       if os.path.isfile(DATA_PATH+'/'+video_name):
                            filenames.append(video_name)
                            labels.append(class_index)

        else:
               video_list_path = f'{ANNO_PATH}/something-something-v2-validation.json' 
               with open(video_list_path) as f:
                   video_infos = json.load(f)
                   for video_info in video_infos:
                       video = int(video_info['id'])
                       video_name = f'{video}.avi'
                       class_name = video_info['template'].replace('[', '').replace(']', '')
                       class_index = int(self.class_idx_dict[class_name])
                       if os.path.isfile(DATA_PATH+'/'+video_name):
                             filenames.append(video_name)
                             labels.append(class_index)

        print(filenames[0:10],labels[0:10])

        if 'train' in subset and num_of_examples!=0:
                  filenames, labels = get_subset_data(filenames,labels,num_of_examples)

        self.num_videos = len(filenames)

        super(SOMETHING, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_clip_duration=video_clip_duration,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=False,
            return_labels=return_labels,
            labels=labels,
            max_offsync_augm=max_offsync_augm,
            mode=mode,
            clips_per_video=clips_per_video,
        )

