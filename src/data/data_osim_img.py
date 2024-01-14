from data.dataset import DatasetBase
import numpy as np

import os
import glob
import cv2
import copy

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
    # (F, N, 2)
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
        
        #mask = cv2.imread(mask_name, 0)
        #mask_view0 = mask/255
        mask_view0 = masks[0]
        ys, xs = np.where(mask_view0 >0)
        pos = np.concatenate((xs[..., None], ys[..., None]), axis=1)
        sample_idx = np.random.choice(np.arange(pos.shape[0]), num_points, replace=False)
        pos_sampled.append(pos[sample_idx])
        
        #mask = cv2.imread(mask_name.replace('view0', 'view1'), 0)
        #mask_view1 = mask/255
        mask_view1 = masks[1]
        ys, xs = np.where(mask_view1 >0)
        pos = np.concatenate((xs[..., None], ys[..., None]), axis=1)
        sample_idx = np.random.choice(np.arange(pos.shape[0]), num_points, replace=False)
        pos_sampled.append(pos[sample_idx])
        
    else:
        #mask = cv2.imread(mask_name, 0)
        #mask_view0 = mask/255
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

        # read dataset
        self.get_data()
        self._dataset_size = len(self.renders)
            
        self.random = np.random.seed(0)

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        clip_name = str.split(self.renders[index], '/')[-4]
        subject = str.split(clip_name, '_')[1]
        action = str.split(clip_name, '_')[2]
        post_fix = str.split(clip_name, '_')[3]
        render_name = os.path.splitext(self.renders[index])[0]
        name_scan =  str.split(render_name, '/')[-1]

        ##################
        #  Prepare image
        ##################
        imgs = []
        masks = []
        meta_data = []
        for i in range(self.num_view):
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
            meta_data.append(np.array(np.load(meta_name, allow_pickle=True)))
            

        #(C, H, W)
        imgs = np.concatenate(imgs, axis=0)
        
        sample = {'input_image': imgs}
        sample['input_meta'] = np.stack(meta_data, axis=0)
        meta_data = sample['input_meta']

        mask_name = self.renders[index].replace('crop', 'mask')
        sample['input_points'] = get_points_from_mask(self.num_view, masks, num_points = self.num_points)

        
        ##########################################
        #  Prepare joint points ground truth
        ##########################################
        osim_coords = np.load(os.path.join(self.data_dir, 'coords', f'{subject}_{action}_{post_fix}', name_scan + '_coords.npy'))
        osim_coords[10] = copy.deepcopy(osim_coords[9])
        osim_coords[18] = copy.deepcopy(osim_coords[17])
        
        osim_points = np.load(os.path.join(self.data_dir, 'points', f'{subject}_{action}_{post_fix}', name_scan + '_joint_points.npy'))
            
        osim_scales = np.load(os.path.join(self.data_dir, 'scales', subject, 'body_scales.npy'), allow_pickle=True)
        
        
        ##########################################
        #  Prepare camera 
        ##########################################
        cam_proj_inv = []
        cam_proj = []
        cam_extrinsic = []
        if self.num_view > 1:
            for i in range(self.num_view):
                camera_name = os.path.join(os.path.dirname(render_name.replace('crop', 'meta')), 'camera_info.npy')
                camera_name = camera_name.replace('view0', 'view%s'%i)
                cam_info = np.load(camera_name, allow_pickle=True).flatten()[0]
                cam_extrinsic.append(cam_info['extrinsic'])
                cam_proj.append(cam_info['proj'])
                cam_proj_inv.append(cam_info['proj_inv'])
                
        else:
            camera_name = os.path.join(os.path.dirname(render_name.replace('crop', 'meta')), 'camera_info.npy')
            cam_info = np.load(camera_name, allow_pickle=True).flatten()[0]
            cam_extrinsic.append(cam_info['extrinsic'])
            cam_proj.append(cam_info['proj'])
            cam_proj_inv.append(cam_info['proj_inv'])
            
        
        # pack data
        sample['target_coords'] = osim_coords[..., None]
        sample['target_points'] = osim_points
        sample['target_scales'] = osim_scales
        sample['input_cam_proj_inv'] = np.stack(cam_proj_inv, axis=0)
        sample['input_cam_proj'] = np.stack(cam_proj, axis=0)
        sample['input_cam_ext'] = np.stack(cam_extrinsic, axis=0)
        
        # (v, 250, 2)
        sample['input_points_full'] = np.empty(sample['input_points'].shape)
        v, n, p = sample['input_points'].shape
        for v in range(sample['input_points'].shape[0]):
            # (1, 1)
            scales = meta_data[v, 2][..., None][..., None]
            # (1)
            offset_x = meta_data[v, 0][..., None]
            # (1)
            offset_y = meta_data[v, 1][..., None]
            # (N, 2)
            sample['input_points_full'][v, :, :] = sample['input_points'][v, :, :] * scales
            
            # (N, 2)
            sample['input_points_full'][v, :, 0] += offset_x
            sample['input_points_full'][v, :, 1] += offset_y
            
            
        # (1, N, 3)
        front_3d_all = proj_2d_to_3d(sample['input_points_full'][0, :, :][None], sample['input_cam_ext'][0, :, :], sample['input_cam_proj_inv'][0, :, :])
        side_3d_all = proj_2d_to_3d(sample['input_points_full'][1, :, :][None], sample['input_cam_ext'][1, :, :], sample['input_cam_proj_inv'][1, :, :])

        point_all = np.concatenate((front_3d_all, side_3d_all), axis=1)
        
        # point_2d_front (1, N, 2)
        point_2d_front = proj_3d_to_2d(point_all, sample['input_cam_proj'][0, :, :], meta_data[ 0, :][None]).astype(np.int32)
        point_2d_side = proj_3d_to_2d(point_all, sample['input_cam_proj'][1, :, :], meta_data[1, :][None]).astype(np.int32)
        
        # check points within mask
        front_within_mask = masks[0][list(point_2d_front[0, :, 1]), list(point_2d_front[0, :, 0])] != 0
        side_within_mask = masks[1][list(point_2d_side[0, :, 1]), list(point_2d_side[0, :, 0])] != 0

        both_within_mask = front_within_mask & side_within_mask
        candidate_points = point_all[:, both_within_mask, :]
        
        # make sure to have enough points
        if candidate_points.shape[1] < self.num_points:
            diff_num = self.num_points - candidate_points.shape[1]
            repeat_idx = np.random.choice(np.arange(candidate_points.shape[1]), diff_num, replace=True)
            candidate_points = np.concatenate((candidate_points, candidate_points[repeat_idx]), axis=1)
            
        sample_idx_ = np.random.choice(np.arange(candidate_points.shape[1]), self.num_points, replace=False)
        point_all = candidate_points[:, sample_idx_]
        
        # reproject
        point_2d_front = proj_3d_to_2d(point_all, sample['input_cam_proj'][0, :, :], meta_data[ 0, :][None]).astype(np.int32)
        point_2d_side = proj_3d_to_2d(point_all, sample['input_cam_proj'][1, :, :], meta_data[1, :][None]).astype(np.int32)
        
        # center to pelvis (mean)
        point_all -= point_all.mean(axis=1, keepdims=True)
        
        
        sample['input_points'] = np.concatenate((point_2d_front, point_2d_side), axis=0)
        sample['input_points_full'] = point_all[0] / 1000
        
        return sample

    def __len__(self):
        return self._dataset_size

    def get_data(self):
        clip_paths = glob.glob(os.path.join(self.data_dir, 'images', '*'))
        self.renders = []
        for clip_path in clip_paths:
            # get image list
            clip_name = str.split(clip_path, '/')[-1]
            subject = str.split(clip_name, '_')[1]
        
            if len(self.data_subjects) != 0 and subject[-4:] not in self.data_subjects:
                continue
            
            self.renders += glob.glob(os.path.join(clip_path, 'view0', 'crop', '*.png'))
            
        self.renders.sort()