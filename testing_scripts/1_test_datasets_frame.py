import os
import sys
current_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_pwd}/../src/')

import glob
from options.test_options import TestOptions
from models.models import ModelsFactory
import os
import cv2
import tqdm
import numpy as np
import torch
import copy

from utils.OSIM import *
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image


coord2idx = {'pelvis_tilt': 0, 'pelvis_list': 1, 'pelvis_rotation': 2, 'pelvis_tx': 3, 
             'pelvis_ty': 4, 'pelvis_tz': 5, 'hip_flexion_r': 6, 'hip_adduction_r': 7, 
             'hip_rotation_r': 8, 'knee_angle_r': 9, 'knee_angle_r_beta': 10, 'ankle_angle_r': 11, 
             'subtalar_angle_r': 12, 'mtp_angle_r': 13, 'hip_flexion_l': 14, 'hip_adduction_l': 15, 
             'hip_rotation_l': 16, 'knee_angle_l': 17, 'knee_angle_l_beta': 18, 'ankle_angle_l': 19, 
             'subtalar_angle_l': 20, 'mtp_angle_l': 21, 'lumbar_extension': 22, 'lumbar_bending': 23, 
             'lumbar_rotation': 24, 'arm_flex_r': 25, 'arm_add_r': 26, 'arm_rot_r': 27,
             'elbow_flex_r': 28, 'pro_sup_r': 29, 'wrist_flex_r': 30, 'wrist_dev_r': 31,
             'arm_flex_l': 32, 'arm_add_l': 33, 'arm_rot_l': 34, 'elbow_flex_l': 35,
             'pro_sup_l': 36, 'wrist_flex_l': 37, 'wrist_dev_l': 38}


body2idx = {'pelvis': 0, 'femur_r': 1, 'tibia_r': 2, 'patella_r': 3,
             'talus_r': 4, 'calcn_r': 5, 'toes_r': 6, 'femur_l': 7,
             'tibia_l': 8, 'patella_l': 9, 'talus_l': 10, 'calcn_l': 11,
             'toes_l': 12, 'torso': 13, 'humerus_r': 14, 'ulna_r': 15,
             'radius_r': 16, 'hand_r': 17, 'humerus_l': 18, 'ulna_l': 19,
             'radius_l': 20, 'hand_l': 21}


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

def crop_image(img, mask):
    (x,y,w,h) = cv2.boundingRect(np.uint8(mask))
    mask = mask[y:y+h,x:x+w]
    img = img[y:y+h,x:x+w]

    # Prepare new image, with correct size:
    margin = 1.1
    im_size = 256
    
    size = int((max(w, h)*margin)//2)
    new_x = size - w//2
    new_y = size - h//2
    new_img = np.zeros((size*2, size*2, 3))
    new_mask = np.zeros((size*2, size*2))
    new_img[new_y:new_y + h, new_x:new_x+w] = img
    new_mask[new_y:new_y + h, new_x:new_x+w] = mask

    # Resizing cropped and centered image to desired size:
    img = cv2.resize(new_img, (im_size,im_size))
    mask = cv2.resize(new_mask, (im_size,im_size))

    offset_x = x - new_x
    offset_y = y - new_y
    

    return img, mask, (offset_x,offset_y), (size*2)/im_size

def vid2frames(video_file, out_clip_dir, fps_out=60):
    
    # ---------------------
    # Read frames: front and side view
    # ---------------------
    ys = YOLO("yolov8m-seg.pt")
    
    vidcap = cv2.VideoCapture(video_file)
    fps_in = vidcap.get(cv2.CAP_PROP_FPS)

    #fps_out = 60
    assert fps_in == fps_in, 'testing data should all be 60 fps'

    success_front, image_front = vidcap.read()
    
    sampled_idx = []
    frame_idx_in = -1
    frame_cnt = -1
    while success_front == True:
        # check if sample
        frame_idx_in += 1
        ratio = fps_in / fps_out
        frame_idx_out = int(frame_idx_in / ratio)
        
        if frame_idx_out > frame_cnt:
            frame_cnt += 1
            sampled_idx.append(frame_idx_in)
            cv2.imwrite(os.path.join(out_clip_dir, f'frame_{frame_cnt}_org.png'), image_front)
            
            results = ys.predict(image_front, verbose = False, save_conf=True)
            for result in results:
                # iterate results
                if result.boxes is None:
                    continue
                
                if result.masks is None:
                    continue
                
                boxes = result.boxes.cpu().numpy().data   
                masks = result.masks.cpu().numpy().data# get boxes on cpu in numpy
                # find the bod with person with highest score
                # extract classes
                clss = boxes[:, 5]
                # get indices of results where class is 0 (people in COCO)
                people_indices = []
                for i, cls in enumerate(clss):
                    if result.names[cls] == 'person':
                        people_indices.append(i)
                # use these indices to extract the relevant masks
                
                selected_idx = -1
                conf = -1000
                for idx in people_indices: 
                    if boxes[idx, 4] > conf:
                        selected_idx = idx
                        conf = boxes[idx, 4] 
                
                masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
                # rescale masks to original image
                masks = scale_image(masks, result.masks.orig_shape)
                masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)
                        
                mask = (masks[selected_idx]>0.2) * 255
                
                img, mask, (off_x,off_y), scales = crop_image(image_front, mask)
                        
                cv2.imwrite(os.path.join(out_clip_dir, f'frame_{frame_cnt}_mask.png'), mask )
                cv2.imwrite(os.path.join(out_clip_dir, f'frame_{frame_cnt}_crop.png'), img)
                # save meta
                meta = [off_x, off_y, scales]
                np.save(os.path.join(out_clip_dir, f'frame_{frame_cnt}_meta.npy'), meta)
                
        success_front, image_front = vidcap.read()
    
    vidcap.release()
    #pose_mp.close()
    
    return sampled_idx

def wrtieCoords2Mot(filename, input_coords, coord2idx, isRadian):
    # c3d data: reads 3D points (no analog data) and takes off computed data
    # trc header
    header1 = {}
    header1['OrigDataStartFrame'] = 0
    header1['DataRate'] = 60
    header1['coordinates_header'] = 'time\t' + '\t'.join(coord2idx.keys())
    header1['NumFrames'] = input_coords.shape[0]

    version = 'version=%s'%(1)
    nRows = 'nRows=%s'%(input_coords.shape[0])
    nColumns = 'nColumns=%s'%(input_coords.shape[1]+1)
    if isRadian is True:
        inDegrees = 'inDegrees=no'
    else:
        inDegrees = 'inDegrees=yes'

    header_trc = '\n'.join(['Results', version, nRows, nColumns, inDegrees, 'endheader', header1['coordinates_header']])

    
    with open(filename, 'w') as mot_o:
        mot_o.write(header_trc+'\n')
    
        t0 = header1['OrigDataStartFrame'] / header1['DataRate']
        tf = (header1['OrigDataStartFrame'] + header1['NumFrames']-1) / header1['DataRate']
        
        mot_time = np.linspace(t0, tf, num=header1['NumFrames'])
        for n in range(mot_time.size):
            coords = list(input_coords[n].squeeze())
            trc_line = '{t}\t'.format(t=mot_time[n]) + '\t'.join(map(str,coords))
            mot_o.write(trc_line+'\n')


class Test:
    def __init__(self):
        self._opt = TestOptions().parse()
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.src_dir = self._opt.src_dir
        self.dataset = self._opt.dataset
        self.exp_name = self._opt.exp_name
        
        self.video_list = glob.glob(os.path.join(self.src_dir, self.dataset, 'video', '*/*_front.mp4'))

        self.out_dir = os.path.join(self._opt.out_dir, self.dataset)
        os.makedirs(self.out_dir, exist_ok=True)

        self._display_visualizer_test()

            
    def preprocess_frames(self, framefile, src_dir):
        
        filename = os.path.splitext(str.split(framefile, '/')[-1])[0]
        frame_idx = str.split(filename, '_')[1]
        
        img_name = os.path.join(src_dir, f'frame_{frame_idx}_crop.png')
        mask_name = os.path.join(src_dir, f'frame_{frame_idx}_mask.png')
        meta_name = os.path.join(src_dir, f'frame_{frame_idx}_meta.npy')
        imgs = []
        masks = []
        # check if skip current frame
        skip_frame = True
        if self._opt.num_view == 2:
            img_name_view1 = img_name.replace('view0', 'view1')
            if os.path.exists(img_name) and os.path.exists(img_name_view1):
                skip_frame = False
        else:
            if os.path.exists(img_name):
                skip_frame = False
            
        
        meta_info_all = []
        if skip_frame is False:
            for i in range(self._opt.num_view):
                # Load input image and mask:
                if self._opt.num_view > 1 & i == 1:
                    img_filename = img_name.replace('view0', 'view1')
                    img = cv2.imread(img_filename)[:,:,::-1]
                    img = cv2.resize(img, (256, 256))
                    mask_filename = mask_name.replace('view0', 'view1')
                    mask = cv2.imread(mask_filename, 0)
                    mask = cv2.resize(mask, (256, 256))/255.0
                    
                    meta_filename = meta_name.replace('view0', 'view1')
                    meta_info = np.load(meta_filename, allow_pickle=True)
                else:
                    img = cv2.imread(img_name)[:,:,::-1]
                    img = cv2.resize(img, (256, 256))
                    mask = cv2.imread(mask_name, 0)
                    mask = cv2.resize(mask, (256, 256))/255.0
                    meta_info = np.load(meta_name, allow_pickle=True)
                

                # Normalize input image:
                img = img/255.0
                imgtensor = img - [0.485, 0.456, 0.406]
                imgtensor = imgtensor / [0.229, 0.224, 0.225]
                imgtensor = imgtensor.transpose(2, 0, 1)
                # Mask background out:
                imgtensor[:, mask[:,:]==0] = 0
                imgtensor = torch.FloatTensor(imgtensor).to(self.device)

                imgs.append(imgtensor)
                masks.append(mask)
                meta_info_all.append(meta_info)
        
        return skip_frame, imgs, meta_info_all, masks
     
    def set_intpu(self, input):
        self._input_image = input['input_image'].unsqueeze(0).float().to(self.device)
        
        self._input_points = torch.from_numpy(input['input_points'][None]).float().to(self.device)
        self._input_points_full = torch.from_numpy(input['input_points_full'][None]).float().to(self.device)

        return 

    def _display_visualizer_test(self):

        # set model to eval
        self._model.set_eval()
        for clip in self.video_list:
            
            fps_out = 60
            subject = str.split(clip, '/')[-2]
            action = str.split(str.split(clip, '/')[-1], '_')[0]
            
            if os.path.exists(os.path.join(self.out_dir, f'ik_mot_files_{self.exp_name}', subject, action)) is True:
                continue
            
            ## save frames from video ...
            img_out_dir1 = os.path.join(self.src_dir, self.dataset, 'images', subject, action, 'view0')
            
            ## read frames and predict frame features ...
            frame_files = glob.glob(os.path.join(img_out_dir1, '*_org.png'))
            # sort frames
            index = []
            for frame in frame_files:
                index.append(int(str.split(os.path.splitext(str.split(frame, '/')[-1])[0], '_')[-2]))
            ind_sorted = np.argsort(np.array(index))
            frame_files = np.array(frame_files)[ind_sorted]
            n_frames = frame_files.shape[0]
            
            print(f'---------- {self.dataset}-{subject}-{action} ---------- ')
            start_frame = 0
            end_frame = n_frames
            
            cam_proj_inv_all = []
            cam_ext_all = []
            cam_proj_all = []
            cam_file = os.path.join(os.path.dirname(clip), f'{subject}_{action}_camera_info.npy')
                          
                         
            for view in range(self._opt.num_view):
                if view == 1:
                    cam_file = os.path.join(os.path.dirname(clip), f'{subject}_{action}_camera_info.npy')
                    
                camera_info = np.load(cam_file, allow_pickle=True).flatten()[0]
                camera_front = {}
                if view == 1:
                    camera_front['intrinsic']= camera_info['side_camera_matrix']
                    camera_front['extrinsic'] = camera_info['side_camera_rot_trans']
                else:
                    camera_front['intrinsic']= camera_info['front_camera_matrix']
                    camera_front['extrinsic'] = camera_info['front_camera_rot_trans']
                
                camera_front['extrinsic'][:, -1] = camera_front['extrinsic'][:, -1] * 1000
                camera_front['distortion'] = np.zeros((5, ))
                camera_front['proj'] = np.matmul(camera_front['intrinsic'],  camera_front['extrinsic'])
                cam_proj_inv = np.linalg.pinv(camera_front['proj'])
                
            
                cam_proj_inv_all.append(cam_proj_inv) 
                cam_ext_all.append(camera_front['extrinsic']) 
                cam_proj_all.append(camera_front['proj']) 
                
            cam_proj_inv_all = np.stack(cam_proj_inv_all, axis=0)
            cam_ext_all = np.stack(cam_ext_all, axis=0)
            cam_proj_all = np.stack(cam_proj_all, axis=0)
            
            
            empty_frames = []
            if fps_out == 30:
                chunk_features = [None] * n_frames * 2
                end_frame = n_frames * 2 
            else:
                chunk_features = [None] * n_frames
                end_frame = n_frames
            
            for cnt, f_idx in enumerate(tqdm.tqdm(range(start_frame, end_frame))):
                
                if fps_out == 30:
                    if f_idx % 2 == 1:
                        empty_frames.append(f_idx)
                        continue
                    
                    frame = frame_files[f_idx // 2]
                else:
                    frame = frame_files[f_idx]
                
                skip_frame, imgs, metas, masks = self.preprocess_frames(frame, img_out_dir1)
                
                if skip_frame is True:
                    empty_frames.append(f_idx)
                    continue
    
                imgs = torch.cat(imgs, dim=0).to(self.device)
                
                sample = {'input_image': imgs}
                sample['input_cam_proj_inv'] = cam_proj_inv_all
                sample['input_cam_proj'] = cam_proj_all
                sample['input_cam_ext'] = cam_ext_all
                
                # get frame features
                with torch.no_grad():
                    # get new input points
                    frame_idx = str.split(os.path.splitext(str.split(frame, '/')[-1])[0], '_')[1]
                    sample['input_points'] = get_points_from_mask(self._opt.num_view, masks, num_points = self._opt.num_points)

                    ###########
                    sample['input_points_full'] = np.empty(sample['input_points'].shape)
                    v, n, p = sample['input_points'].shape
                    for v in range(sample['input_points'].shape[0]):
                        # (1, 1)
                        scales = metas[v][2][..., None][..., None]
                        # (1)
                        offset_x = metas[v][0][..., None]
                        # (1)
                        offset_y = metas[v][1][..., None]
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
                    point_2d_front = proj_3d_to_2d(point_all, sample['input_cam_proj'][0, :, :], metas[0][None]).astype(np.int32)
                    point_2d_side = proj_3d_to_2d(point_all, sample['input_cam_proj'][1, :, :], metas[1][None]).astype(np.int32)
                    
                    # check points within mask
                    front_within_mask = masks[0][list(point_2d_front[0, :, 1]), list(point_2d_front[0, :, 0])] != 0
                    side_within_mask = masks[1][list(point_2d_side[0, :, 1]), list(point_2d_side[0, :, 0])] != 0

                    both_within_mask = front_within_mask & side_within_mask
                    candidate_points = point_all[:, both_within_mask, :]
                    
                    # make sure to have enough points
                    if candidate_points.shape[1] < self._opt.num_points:
                        diff_num = self._opt.num_points - candidate_points.shape[1]
                        repeat_idx = np.random.choice(np.arange(candidate_points.shape[1]), diff_num, replace=True)
                        candidate_points = np.concatenate((candidate_points, candidate_points[:, repeat_idx]), axis=1)
                        
                    sample_idx_ = np.random.choice(np.arange(candidate_points.shape[1]), self._opt.num_points, replace=False)
                    point_all = candidate_points[:, sample_idx_]
                    
                    # reproject
                    point_2d_front = proj_3d_to_2d(point_all, sample['input_cam_proj'][0, :, :], metas[0][None]).astype(np.int32)
                    point_2d_side = proj_3d_to_2d(point_all, sample['input_cam_proj'][1, :, :], metas[1][None]).astype(np.int32)
                    
                    
                    # center to pelvis (mean)
                    point_all -= point_all.mean(axis=1, keepdims=True)
                    
                    
                    sample['input_points'] = np.concatenate((point_2d_front, point_2d_side), axis=0)
                    sample['input_points_full'] = point_all[0] / 1000
                        
                    ##############################
                    self.set_intpu(sample)
                    self._model._img_encoder(self._input_image)
                    pred = self._model._img_encoder.query_frame(self._input_points, self._input_points_full)
                  
                    chunk_features[cnt] = pred
                        
                    _B = pred.shape[0]
                    pred_angles = pred[:, :36]
                    pred_pelvis_pos = torch.zeros((_B, 3, 1)).float().to(self.device)
                    pred_coords = torch.concatenate((pred_angles[:, :3], pred_pelvis_pos, pred_angles[:, 3:]), dim=1)
                    pred_scales = pred[:, 36:].reshape(_B, -1, 3) / 8 + 1.0
            
            
                    pred_coords_sp = pred_coords[0].squeeze(-1).cpu().data.numpy()
                    pred_scales_sp = pred_scales[0].cpu().data.numpy()
                    
                    sc_filename = os.path.join(self.out_dir, f'osim_model_{self.exp_name}', subject, action, f'osim_scales_{f_idx}_scaled.npy')
                    os.makedirs(os.path.dirname(sc_filename), exist_ok=True)
                    np.save(sc_filename, pred_scales_sp)
                    
                    out_mot_file = os.path.join(self.out_dir, f'ik_mot_files_{self.exp_name}', subject, action, f'ik_{f_idx}_radians.mot')
                    os.makedirs(os.path.dirname(out_mot_file), exist_ok=True)
                    wrtieCoords2Mot(out_mot_file, pred_coords_sp[None], coord2idx, True)
                    
                
            empty_filename = os.path.join(self.out_dir, f'ik_mot_files_{self.exp_name}', subject, action, f'interpolated_frames.npy')        
            os.makedirs(os.path.dirname(empty_filename), exist_ok=True)
            np.save(empty_filename, empty_frames)
            
            # interpolate missing frames
            prev_feat_idx = -1
            interpolate_list = []
            for f_i, feat in enumerate(chunk_features):
                if feat is not None:
                    prev_feat_idx = f_i
                else:
                    # find the closet feature after
                    feat_ = feat
                    cnt = 1
                    first_feat_idx = -1
                    while feat_ is None and f_i+cnt < len(chunk_features):
                        feat_ = chunk_features[f_i+cnt]
                        cnt += 1
                    
                    if feat_ is not None:
                        first_feat_idx = f_i+cnt-1
                        
                    interpolate_list.append([f_i, prev_feat_idx, first_feat_idx])
                    
            # interpolate
            skip_chunk = False
            for info in interpolate_list:
                if info[1] == -1 and info[2] == -1:
                    skip_chunk = True
                    break
                elif info[1] != -1 and info[2] != -1:
                    step_size = info[2] - info[1]
                    chunk_features[info[0]] = chunk_features[info[1]] * ((info[2] - info[0])/step_size) + chunk_features[info[2]]  * ((info[0] - info[1])/step_size)
                elif info[1] == -1:
                    chunk_features[info[0]] = chunk_features[info[2]]
                elif info[2] == -1:
                    chunk_features[info[0]] = chunk_features[info[1]]
                    
                pred = chunk_features[info[0]]
                _B = pred.shape[0]
                pred_angles = pred[:, :36]
                pred_pelvis_pos = torch.zeros((_B, 3, 1)).float().to(self.device)
                pred_coords = torch.concatenate((pred_angles[:, :3], pred_pelvis_pos, pred_angles[:, 3:]), dim=1)
                pred_scales = pred[:, 36:].reshape(_B, -1, 3) / 8 + 1.0
            
                pred_coords_sp = pred_coords[0].squeeze(-1).cpu().data.numpy()
                pred_scales_sp = pred_scales[0].cpu().data.numpy()
                
                sc_filename = os.path.join(self.out_dir, f'osim_model_{self.exp_name}', subject, action, f'osim_scales_{start_frame + info[0]}_scaled.npy')
                os.makedirs(os.path.dirname(sc_filename), exist_ok=True)
                np.save(sc_filename, pred_scales_sp)
                
                out_mot_file = os.path.join(self.out_dir, f'ik_mot_files_{self.exp_name}', subject, action, f'ik_{start_frame + info[0]}_radians.mot')
                os.makedirs(os.path.dirname(out_mot_file), exist_ok=True)
                wrtieCoords2Mot(out_mot_file, pred_coords_sp[None], coord2idx, True)
             
            if skip_chunk is False:
                
                pred_coords_sp_all = []
                pred_scales_sp_all = []
                for f_idx, pred in enumerate(chunk_features):
                    pred_angles = pred[:, :36]
                    pred_pelvis_pos = torch.zeros((_B, 3, 1)).float().to(self.device)
                    pred_coords_sp = torch.concatenate((pred_angles[:, :3], pred_pelvis_pos, pred_angles[:, 3:]), dim=1)
                    pred_scales_sp = pred[:, 36:].reshape(_B, -1, 3) / 8 + 1.0
            
                    pred_coords_sp = pred_coords_sp[0].squeeze(-1).cpu().data.numpy()
                    pred_coords_sp_all.append(pred_coords_sp)
                    
                    pred_scales_sp = pred_scales_sp[0].cpu().data.numpy()
                    pred_scales_sp_all.append(pred_scales_sp)
                    
                pred_coords_sp_all = np.stack(pred_coords_sp_all)
                pred_scales_sp_all = np.stack(pred_scales_sp_all)
                
                out_mot_file = os.path.join(self.out_dir, f'ik_mot_files_{self.exp_name}', subject, f'ik_{subject}_{action}_all_radians.mot')
                os.makedirs(os.path.dirname(out_mot_file), exist_ok=True)
                wrtieCoords2Mot(out_mot_file, pred_coords_sp_all, coord2idx, True)
                
            else:
                print(f'Failed to detect any pose from {clip}')
                        
                    


if __name__ == "__main__":
    
    Test()
