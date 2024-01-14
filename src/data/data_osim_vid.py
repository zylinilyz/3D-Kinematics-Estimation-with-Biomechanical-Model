from data.dataset import DatasetBase
import numpy as np

import os
import glob
import cv2

#from utils.util_opensim_wo_lib import *
from utils.generators import *
import copy
from os import path

def project_by_camera_projection(joints_cv, pro_side):
    """
    Reference: https://github.com/google-research-datasets/Objectron/blob/master/notebooks/objectron-geometry-tutorial.ipynb
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    function project_points
    """
    vertices_3d = joints_cv.reshape(joints_cv.shape[0], -1,3)
    vertices_3d_homg = np.concatenate((vertices_3d, np.ones_like(vertices_3d[:, :, :1])), axis=-1).transpose((0, 2, 1))
    # vertices_2d_proj (F, 3, 1)
    vertices_2d_proj = np.matmul(pro_side[None], vertices_3d_homg)
    # Project the points
    points2d_ndc = vertices_2d_proj[:, :-1, :] / vertices_2d_proj[:, -1:, :]
    points2d_ndc = points2d_ndc.transpose((0, 2, 1))
    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    arranged_points = points2d_ndc[:, :,:2]
    return arranged_points

def proj_3d_to_2d(point_3d, pro_front, meta_front):
    # F, 500, 3
    projected_2d = project_by_camera_projection(point_3d, pro_front)

    # (F, 1, 1)
    scales = meta_front[:, 2][..., None][..., None]
    # (F, 1)
    offset_x = meta_front[:, 0][..., None]
    offset_y = meta_front[:, 1][..., None]

    points_2d = copy.deepcopy(projected_2d)
    points_2d[:, :, 0] -= offset_x
    points_2d[:, :, 1] -= offset_y
    points_2d = points_2d / scales

    points_2d = np.clip(points_2d, 0, 255)
    
    return points_2d

def proj_2d_to_3d(front_2d, front_ext, inv_pro_front):
    
    front_2d = np.concatenate((front_2d,  np.ones((front_2d.shape[0], front_2d.shape[1], 1))), axis=-1)
    # inv_pro_front(1, 1, 4, 3) front_2d(f, N, 3, 1)
    front_3d = np.matmul(inv_pro_front[None][None], front_2d[...,None])
    front_3d[:, :, :-1] /= front_3d[:, :, -1:]

    # Transform camera origin in World coordinate frame
    # (1, 1, 3)
    cam_world = - np.matmul(front_ext[:3, :3].T, front_ext[:, -1])
    cam_world = cam_world[None][None]
    vector = front_3d[:, :, :3, 0] - cam_world
    # (f, N, 1)
    org_length = np.linalg.norm(vector, axis=-1, keepdims=True)
    unit_vector = vector / org_length

    # Point scaled along this ray
    # sample along ray
    front_3d_all = []
    for i in range(-int(org_length.mean()/2), int(org_length.mean()/2), 50):
        front_3d_all.append(cam_world + (org_length-i) * unit_vector)
    front_3d_all = np.concatenate(front_3d_all, axis=1)
    
    return front_3d_all

def get_points_from_mask(num_view, masks, num_points=6890):
    pos_sampled = []
    if num_view > 1:
        
        mask_view0 = masks[0]
        ys, xs = np.where(mask_view0 >0)
        pos = np.concatenate((xs[..., None], ys[..., None]), axis=1)
        sample_idx = np.random.choice(np.arange(pos.shape[0]), num_points, replace=False)
        pos_sampled.append(pos[sample_idx])
        
        mask_view1 = masks[1]
        ys, xs = np.where(mask_view1 >0)
        pos = np.concatenate((xs[..., None], ys[..., None]), axis=1)
        sample_idx = np.random.choice(np.arange(pos.shape[0]), num_points, replace=False)
        pos_sampled.append(pos[sample_idx])
        
    else:
        mask_view0 = masks[0]
        ys, xs = np.where(mask_view0 >0)
        pos = np.concatenate((xs[..., None], ys[..., None]), axis=1)
        sample_idx = np.random.choice(np.arange(pos.shape[0]), num_points, replace=False)
        pos_sampled.append(pos[sample_idx])
        
    pos_sampled = np.stack(pos_sampled, axis=0)
    
    return pos_sampled

class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self._name = 'Dataset'
        self.data_dir = opt.train_data_dirname
        self.data_subjects = []
           
        if self._mode == 'val':
            self.data_dir = opt.train_data_dirname.replace('train_data_yolo', 'val_data_yolo')
            
        self.num_view = opt.num_view
        self.num_points = opt.num_points
        self.batch_size = opt.batch_size
        self.chunk_length = opt.chunk_length

        # read dataset
        self.get_data()

        n_chunks_tot = len(self.generator.pairs)
        self._dataset_size = n_chunks_tot

    def __getitem__(self, index):
        #print('index', index)
        assert (index < self._dataset_size)


        seq_i, start_f, end_f = self.generator.pairs[index]
        #print('---------------', seq_i, start_f, end_f)
        images, coords, scale_file, points = self.generator.get_chunk(seq_i, start_f, end_f)
        
        sample = {}
        seq_imgs = []
        seq_in_points = []
        seq_coords = []
        seq_out_points = []
        seq_meta = []
        seq_cam_extrinsic = []
        seq_cam_proj_inv = []
        seq_cam_proj = []
        seq_res_x = []
        seq_res_y = []
 
        ##########################################
        #  Prepare camera 
        ##########################################
        if self.num_view > 1:
            for i in range(self.num_view):
                camera_name = os.path.join(os.path.dirname(images[0].replace('crop', 'meta')), 'camera_info.npy')
                camera_name = camera_name.replace('view0', 'view%s'%i)
                cam_info = np.load(camera_name, allow_pickle=True).flatten()[0]
                seq_cam_extrinsic.append(cam_info['extrinsic'])
                seq_res_x.append(cam_info['intrinsic'][0, 2])
                seq_res_y.append(cam_info['intrinsic'][1, 2])
                seq_cam_proj.append(cam_info['proj'])
                seq_cam_proj_inv.append(cam_info['proj_inv'])
                
        else:
            camera_name = os.path.join(os.path.dirname(images[0].replace('crop', 'meta')), 'camera_info.npy')
            cam_info = np.load(camera_name, allow_pickle=True).flatten()[0]
            seq_cam_extrinsic.append(cam_info['extrinsic'])
            seq_res_x.append(cam_info['intrinsic'][0, 2])
            seq_res_y.append(cam_info['intrinsic'][1, 2])
            seq_cam_proj.append(cam_info['proj'])
            seq_cam_proj_inv.append(cam_info['proj_inv'])
        
        seq_res_x = np.stack(seq_res_x, axis=0)
        seq_res_y = np.stack(seq_res_y, axis=0)
        
        # loop through frames
        for frame, coord, point in zip(images, coords, points):
            
            render_name = os.path.splitext(frame)[0]
            name_scan =  str.split(render_name, '/')[-1]
    
            ##################
            #  Prepare image
            ##################
            imgs = []
            masks = []
            meta_data = []
            for i in range(self.num_view):
                #print('##############', i)
                if self.num_view > 1:
                    img_name = render_name.replace('view0', 'view%s'%i) + '.png'
                    mask_name = img_name.replace('crop', 'mask')
                    meta_name = render_name.replace('view0', 'view%s'%i).replace('crop', 'meta') + '.npy'
                else:
                    img_name = render_name + '.png'
                    mask_name = img_name.replace('crop', 'mask')
                    meta_name = render_name.replace('crop', 'meta') + '.npy'
                    
                img = cv2.imread(img_name)[:,:,::-1]
                img = cv2.resize(img, (256, 256))
                mask = cv2.imread(mask_name, 0)
                mask = cv2.resize(mask, (256, 256))/255.0
                
                masks.append(mask)
    
                # Normalize input image:
                img = img/255.0
                img = img - [0.485, 0.456, 0.406]
                img = img / [0.229, 0.224, 0.225]
                img = img.transpose(2, 0, 1)
                img[:, mask==0] = 0
                mask = mask[...,None]
                
                #(C, H, W)
                img = np.concatenate((img, mask.transpose(2, 0, 1)))
    
                imgs.append(img)
                
                ##########################################
                #  Prepare meta 
                ##########################################
                meta_data.append(np.array(np.load(meta_name, allow_pickle=True)))
                
                
    
            #(C, H, W)
            imgs = np.concatenate(imgs, axis=0)
            meta_data = np.stack(meta_data, axis=0)
            
            seq_imgs.append(imgs)
            seq_meta.append(meta_data)
    
            seq_in_points.append(get_points_from_mask(self.num_view, masks, num_points = self.num_points))
                
            
            ##########################################
            #  Prepare coordinates ad points
            ##########################################
            osim_coords = np.load(coord)
            osim_coords[10] = copy.deepcopy(osim_coords[9])
            osim_coords[18] = copy.deepcopy(osim_coords[17])
            seq_coords.append(osim_coords)
    
            # points
            seq_out_points.append(np.load(point))
            
            
        # scale
        osim_scales = np.load(scale_file, allow_pickle=True)
        
        # F, 4*2, 256, 256
        sample['input_image'] = np.stack(seq_imgs, axis=0)
        sample['input_points'] = np.stack(seq_in_points, axis=0)

        sample['target_scales'] = osim_scales
        
        sample['target_coords'] = np.stack(seq_coords, axis=0)[..., None]
        sample['target_points'] = np.stack(seq_out_points, axis=0)
        
        sample['input_cam_proj_inv'] = np.stack(seq_cam_proj_inv, axis=0)
        sample['input_cam_ext'] = np.stack(seq_cam_extrinsic, axis=0)
        sample['input_cam_proj'] = np.stack(seq_cam_proj, axis=0)
        
        sample['input_meta'] = np.stack(seq_meta, axis=0)
        input_meta = sample['input_meta']
        # get input point_full
        # (f, v, 250, 2)
        f, v, n, p = sample['input_points'].shape
        sample['input_points_full'] = np.empty((f, v, n, 2))
        for v in range(sample['input_points'].shape[1]):
            # (F, 1, 1)
            scales = input_meta[:, v, 2][..., None][..., None]
            # (F, 1)
            offset_x = input_meta[:, v, 0][..., None]
            # (F, 1)
            offset_y = input_meta[:, v, 1][..., None]
            # (F, N, 2)
            sample['input_points_full'][:, v, :, :] = sample['input_points'][:, v, :, :] * scales
            
            # (F, N, 2)
            sample['input_points_full'][:, v, :, 0] += offset_x
            sample['input_points_full'][:, v, :, 1] += offset_y
        
        # (F, N, 3)
        front_3d_all = proj_2d_to_3d(sample['input_points_full'][:, 0, :, :], sample['input_cam_ext'][0, :, :], sample['input_cam_proj_inv'][0, :, :])
        side_3d_all = proj_2d_to_3d(sample['input_points_full'][:, 1, :, :], sample['input_cam_ext'][1, :, :], sample['input_cam_proj_inv'][1, :, :])

        point_all = np.concatenate((front_3d_all, side_3d_all), axis=1)

        # point_2d_front (F, N, 2)
        point_2d_front = proj_3d_to_2d(point_all, sample['input_cam_proj'][0, :, :], input_meta[:, 0, :]).astype(np.int32)
        point_2d_side = proj_3d_to_2d(point_all, sample['input_cam_proj'][1, :, :], input_meta[:, 1, :]).astype(np.int32)
        
        # check points within mask
        mask0 = sample['input_image'][:, 3, :, :]
        mask1 = sample['input_image'][:, 7, :, :]
        F = mask1.shape[0]
        point_all_new = []
        for f_i in range(F):
            front_within_mask = mask0[f_i][list(point_2d_front[f_i, :, 1]), list(point_2d_front[f_i, :, 0])] != 0
            side_within_mask = mask1[f_i][list(point_2d_side[f_i, :, 1]), list(point_2d_side[f_i, :, 0])] != 0

            both_within_mask = front_within_mask & side_within_mask
            candidate_points = point_all[f_i, both_within_mask, :][None]
            
            # make sure to have enough points
            if candidate_points.shape[1] < self.num_points:
                diff_num = self.num_points - candidate_points.shape[1]
                repeat_idx = np.random.choice(np.arange(candidate_points.shape[1]), diff_num, replace=True)
                candidate_points = np.concatenate((candidate_points, candidate_points[repeat_idx]), axis=1)
                
            sample_idx_ = np.random.choice(np.arange(candidate_points.shape[1]), self.num_points, replace=False)
            point_all_new.append(candidate_points[:, sample_idx_])
        
        point_all = np.concatenate(point_all_new, axis=0)
        # reproject
        point_2d_front = proj_3d_to_2d(point_all, sample['input_cam_proj'][0, :, :], input_meta[:, 0, :]).astype(np.int32)
        point_2d_side = proj_3d_to_2d(point_all, sample['input_cam_proj'][1, :, :], input_meta[:,1, :]).astype(np.int32)
        
        # center to pelvis (mean)
        point_all -= point_all.mean(axis=1, keepdims=True)
        
        
        sample['input_points'] = np.stack((point_2d_front, point_2d_side), axis=1)
        sample['input_points_full'] = point_all / 1000
            
        return sample

    def __len__(self):
        return self._dataset_size
        
    def get_data(self):
        # get all clips' name
        clip_paths = glob.glob(path.join(self.data_dir, 'images', '*'))
        
        # loop through each clip
        self.clip_frames = []
        self.clip_coords = []
        self.clip_points = []
        self.clip_scales = []
        for clip_path in clip_paths:
            # get image list
            clip_name = str.split(clip_path, '/')[-1]
            subject = str.split(clip_name, '_')[1]
            if len(self.data_subjects) != 0 and subject[-4:] not in self.data_subjects:
                continue
            action = str.split(clip_name, '_')[2]
            post_fix = str.split(clip_name, '_')[3]
            
            frames = glob.glob(path.join(clip_path, 'view0', 'crop', '*'))
            index = []
            for frame in frames:
                index.append(int(str.split(os.path.splitext(str.split(frame, '/')[-1])[0], '_')[-1]))
            ind_sorted = np.argsort(np.array(index))

            frames_sorted = np.array(frames)[ind_sorted]
            #print(frames_sorted)
            self.clip_frames.append(frames_sorted)
            # get coords
            coords_name = []
            #/Users/zylin/OneDrive - Delft University of Technology/master_thesis/data_opensim/synthetic_data_0421/train_data/coords/clip_P1108_3_C501
            coords_path = path.join(self.data_dir, 'coords', f'{subject}_{action}_{post_fix}')
            for frame in frames_sorted:
                frame_name = os.path.splitext(str.split(frame, '/')[-1])[0]
                coords_name.append(path.join(coords_path, frame_name + '_coords.npy'))

            self.clip_coords.append(np.array(coords_name))

            # get scales
            scales_path = path.join(self.data_dir, 'scales', subject)
            scales_name = path.join(scales_path, 'body_scales.npy')
            
            self.clip_scales.append(scales_name)

            # get points
            points_name = []
            points_path = path.join(self.data_dir, 'points', f'{subject}_{action}_{post_fix}')
            for frame in frames_sorted:
                frame_name = os.path.splitext(str.split(frame, '/')[-1])[0]
                points_name.append(path.join(points_path, frame_name + '_joint_points.npy'))

            self.clip_points.append(np.array(points_name))
            

        # (n_clip, n_frames)
        # target: (clip_i, start, end, target)
        self.generator = ChunkedGenerator(self.clip_frames, self.clip_coords
                                          , self.clip_points, self.clip_scales
                                          , chunk_length = self.chunk_length )