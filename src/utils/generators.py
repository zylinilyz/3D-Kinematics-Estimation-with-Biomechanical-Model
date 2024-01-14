# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np


class ChunkedGenerator:
    def __init__(self, clip_frames, clip_coords
                     , clip_points, clip_scales, chunk_length=1):
        '''
        batch_size: in chunks

        '''
        self.clip_frames = clip_frames
        self.clip_coords = clip_coords
        self.clip_points = clip_points
        self.clip_scales = clip_scales

        self.chunk_length = chunk_length
        #self.random = np.random.RandomState(random_seed)

        self.gen_pairs()

    # recontruct pairs
    def gen_pairs(self):
        self.pairs = []
        for clip_i in range(len(self.clip_frames)):
            offsets = np.arange(0, self.chunk_length, int(self.chunk_length//2))
            for offset in offsets:
                n_frames = self.clip_frames[clip_i].shape[0] - offset
                n_chunks = n_frames // self.chunk_length
                bounds = offset + np.arange(n_chunks + 1) * self.chunk_length
                self.pairs += zip(np.repeat(clip_i, len(bounds)-1), bounds[:-1], bounds[1:])
                
        #for clip_i in range(len(self.clip_frames)):
        #    n_frames = self.clip_frames[clip_i].shape[0]
        #    n_chunks = n_frames // self.chunk_length
        #    bounds = np.arange(n_chunks + 1) * self.chunk_length
        #    self.pairs += zip(np.repeat(clip_i, len(bounds)-1), bounds[:-1], bounds[1:])

    def get_chunk(self, seq_i, start_f, end_f):
        
        out_frames = self.clip_frames[seq_i][start_f:end_f]
        out_coords = self.clip_coords[seq_i][start_f:end_f]
        out_scales = self.clip_scales[seq_i]
        out_points = self.clip_points[seq_i][start_f:end_f]
        
        return out_frames, out_coords, out_scales, out_points



class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d