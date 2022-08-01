# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import glob
import numpy as np


DATA_PATH = '/local-ssd/fmthoker/kinetics/VideoData'
ANNO_PATH = '/local-ssd/fmthoker/kinetics/labels'


from datasets.video_db import VideoDataset
class Kinetics(VideoDataset):
    def __init__(self, subset,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=1,
                 num_of_examples = 0,
                 ):

        classes = sorted(os.listdir(f"{DATA_PATH}"))

        filenames = []
        labels = []
        if  'train' in subset:

               for ln in open(f'{ANNO_PATH}/train_videofolder.txt'):
                      file_name, label = ln.strip().split()[0]+'.avi',int(ln.strip().split()[2])
                      if os.path.isfile(DATA_PATH+'/'+file_name):
                              #print(file_name,label)
                              filenames.append(file_name)
                              labels.append(label)

        else:
               for ln in open(f'{ANNO_PATH}/val_videofolder.txt'):
                      file_name, label = ln.strip().split()[0]+'.avi',int(ln.strip().split()[2])
                      if os.path.isfile(DATA_PATH+'/'+file_name):
                              filenames.append(file_name)
                              labels.append(label)
        print(filenames[0:10],labels[0:10])


        super(Kinetics, self).__init__(
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

        self.name = 'Kinetics dataset'
        self.root = DATA_PATH
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])
