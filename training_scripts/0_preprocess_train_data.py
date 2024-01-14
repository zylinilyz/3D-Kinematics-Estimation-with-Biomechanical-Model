############################
# From video to frames (org, crop, mask)
############################

import glob
import os
import sys
current_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_pwd}/../src/')
OSIM_MODEL_PATH = f'{current_pwd}/../src/utils/opensim_models/osim_model_scale1.npy'
DEFAULT_DATA_PATH = f'{current_pwd}/../data/train_data/'

import cv2
import numpy as np
from bs4 import BeautifulSoup
from scipy.spatial.transform import Rotation as R
from utils.OSIM import *
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

def read_mot(filename, coord2idx):
    
    # Using readlines()
    file1 = open(filename, 'r')
    
    # Strips the newline character
    extract = False
    coords_arr = []
    for line in file1.readlines():
        #print(line)
        if extract:
            coords = str.split(line.strip(), '\t')[1:]
            coords_tmp = np.ones(len(coords))
           
            for c in range(len(coords)):
                coords_tmp[coord2idx[coord_keys[c]]] = float(coords[c])
    
            coords_arr.append(list(coords_tmp))
    
        if str.split(line.strip(), '\t')[0] == 'time':
            coord_keys = str.split(line.strip(), '\t')[1:]
            extract = True
    
    coords_arr = np.array(coords_arr, dtype=np.float64)

    return coords_arr, coord_keys

def crop_image(img, mask):
    (x,y,w,h) = cv2.boundingRect(np.uint8(mask))
    mask = mask[y:y+h,x:x+w]
    img = img[y:y+h,x:x+w]

    # Prepare new image, with correct size:
    margin = 1.1
    im_size = 256
    clean_im_size = im_size/margin
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

def create_dirs(out_dir):
    org_dir = os.path.join(out_dir, 'org')
    os.makedirs(org_dir, exist_ok=True)
    crop_dir = os.path.join(out_dir, 'crop')
    os.makedirs(crop_dir, exist_ok=True)
    mask_dir = os.path.join(out_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    meta_dir = os.path.join(out_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)

    return org_dir, crop_dir, mask_dir, meta_dir


def wrtieCoords2Mot(filename, input_coords, coord2idx, isRadian, fps = 60):
    # c3d data: reads 3D points (no analog data) and takes off computed data
    # trc header
    header1 = {}
    header1['OrigDataStartFrame'] = 0
    header1['DataRate'] = fps
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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_PATH, help="Specify data_dir")    
    parser.add_argument('--out_dir', type=str, default=DEFAULT_DATA_PATH, help="Specify out_dir")   
    
    args = parser.parse_args()    
    
    video_list = glob.glob(os.path.join(args.data_dir, 'video', '*/*/*_front.mp4'))

    img_out_dir = os.path.join(args.out_dir, 'train_data_yolo', 'images')
    coords_out_dir = os.path.join(args.out_dir, 'train_data_yolo', 'coords')
    points_out_dir = os.path.join(args.out_dir, 'train_data_yolo', 'points')
    scales_out_dir = os.path.join(args.out_dir, 'train_data_yolo', 'scales')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(coords_out_dir, exist_ok=True)
    os.makedirs(points_out_dir, exist_ok=True)
    os.makedirs(scales_out_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bmi_list = ['bmi-1.0', 'bmi-0.5', 'bmiORI', 'bmi0.5', 'bmi1.0']
    # loop over all clips
    for vid_idx, video_file in enumerate(video_list):
        ys = YOLO("yolov8m-seg.pt")
        
        drop_video = False
        subject = str.split(video_file, '/')[-3]
        bmi_idx = bmi_list.index(str.split(video_file, '/')[-2])
        action = str.split(str.split(video_file, '/')[-1], '_')[1]
        
        if os.path.exists(os.path.join(points_out_dir, f'{subject}_{action}_{bmi_idx}')) is True:
            continue

        out_clip_dir = os.path.join(img_out_dir, f'clip_{subject}_{action}_{bmi_idx}')
        os.makedirs(out_clip_dir, exist_ok=True)
    
        video_front = video_file
        out_front_dir = os.path.join(out_clip_dir, 'view0')
        os.makedirs(out_front_dir, exist_ok=True)
    
        video_side = video_file.replace('_front', '_side')
        out_side_dir = os.path.join(out_clip_dir, 'view1')
        os.makedirs(out_side_dir, exist_ok=True)
        
        # ---------------------
        # Read frames: front and side view
        # ---------------------
        vidcap_front = cv2.VideoCapture(video_front)
        vidcap_side = cv2.VideoCapture(video_side)

        fps_in = vidcap_front.get(cv2.CAP_PROP_FPS)
        assert fps_in == vidcap_side.get(cv2.CAP_PROP_FPS), 'Frame rate mismatch!'
        
        # target output fps
        fps_out = 60

        index_in = -1
        index_out = -1

        success_front, image_front = vidcap_front.read()
        success_side, image_side = vidcap_side.read()
        
        sampled_idx = []
        frame_idx_in = -1
        frame_cnt = -1
        first_frame_idx = frame_cnt
        while success_front == True & success_side == True:
            # check if sample
            frame_idx_in += 1
            ratio = fps_in / fps_out
            frame_idx_out = int(frame_idx_in / ratio)

            if frame_idx_out > frame_cnt:
                frame_cnt += 1
                
                found_mask_front = False
                results = ys.predict(image_front[:,:,::-1], verbose = False)
                for result in results:
                    # iterate results
                    if result.boxes is not None and result.masks is not None:
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
                                
                        mask_front = (masks[selected_idx]>0.2) * 255
                        found_mask_front = True
                        
                            
                
                found_mask_side = False
                results = ys.predict(image_side[:,:,::-1], verbose = False)
                for result in results:
                    # iterate results
                    if result.boxes is not None and result.masks is not None:
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
                                
                        mask_side = (masks[selected_idx]>0.2) * 255
                        found_mask_side = True
    
                
                if found_mask_front == False or found_mask_side == False:
                    if len(sampled_idx) == 0:
                        success_front, image_front = vidcap_front.read()
                        success_side, image_side = vidcap_side.read()
                        continue
                    else:
                        break
                
                sampled_idx.append(frame_idx_in) 
                image_front_crop, mask_front, (off_x,off_y), scales = crop_image(image_front, mask_front)
                meta_front = [off_x,off_y, scales]
                image_side_crop, mask_side, (off_x,off_y), scales = crop_image(image_side, mask_side)
                meta_side = [off_x,off_y, scales]

                # record crop position and resize scales
                if first_frame_idx == -1:
                    first_frame_idx = frame_cnt
                org_dir, crop_dir, mask_dir, meta_dir = create_dirs(out_front_dir)
                cv2.imwrite(os.path.join(org_dir, f'frame_{frame_cnt}.png'), image_front)
                cv2.imwrite(os.path.join(crop_dir, f'frame_{frame_cnt}.png'), image_front_crop)
                cv2.imwrite(os.path.join(mask_dir, f'frame_{frame_cnt}.png'), mask_front)
                np.save(os.path.join(meta_dir, f'frame_{frame_cnt}.npy'), meta_front)
            
                org_dir, crop_dir, mask_dir, meta_dir = create_dirs(out_side_dir)
                cv2.imwrite(os.path.join(org_dir, f'frame_{frame_cnt}.png'), image_side)
                cv2.imwrite(os.path.join(crop_dir, f'frame_{frame_cnt}.png'), image_side_crop)
                cv2.imwrite(os.path.join(mask_dir, f'frame_{frame_cnt}.png'), mask_side)
                np.save(os.path.join(meta_dir, f'frame_{frame_cnt}.npy'), meta_side)
                
                    
            success_front, image_front = vidcap_front.read()
            success_side, image_side = vidcap_side.read()
        
        vidcap_front.release()
        vidcap_side.release()
        
        # check if continue
        drop_video = True if len(sampled_idx) < 64 else False
        # check if continue
        if drop_video is True:
            print(f'=============== Drop {video_file} (len: {len(sampled_idx)})===============')
            os.system(f'rm -rf {out_clip_dir}')
            continue
        
        # ---------------------
        # Get camera info
        # ---------------------
        camera_file = os.path.join(os.path.dirname(video_file), f'{subject}_{action}_camera_info.npy')
        camera_info = np.load(camera_file, allow_pickle=True).flatten()[0]
        
        camera_side = {}
        camera_side['intrinsic'] = camera_info['side_camera_matrix']
        camera_side['extrinsic'] = camera_info['side_camera_rot_trans']
        camera_side['extrinsic'][:, -1] = camera_side['extrinsic'][:, -1] * 1000
        camera_side['proj'] = np.matmul(camera_side['intrinsic'],  camera_side['extrinsic'])
        camera_side['proj_inv'] = np.linalg.pinv(camera_side['proj'])
        org_dir, crop_dir, mask_dir, meta_dir = create_dirs(out_side_dir)
        np.save(os.path.join(meta_dir, f'camera_info.npy'), camera_side)

        camera_front = {}
        camera_front['intrinsic']= camera_info['front_camera_matrix']
        camera_front['extrinsic'] = camera_info['front_camera_rot_trans']
        camera_front['extrinsic'][:, -1] = camera_front['extrinsic'][:, -1] * 1000
        camera_front['proj'] = np.matmul(camera_front['intrinsic'],  camera_front['extrinsic'])
        camera_front['proj_inv'] = np.linalg.pinv(camera_front['proj'])
        org_dir, crop_dir, mask_dir, meta_dir = create_dirs(out_front_dir)
        np.save(os.path.join(meta_dir, f'camera_info.npy'), camera_front)

        # ---------------------
        # Get Scales
        # ---------------------
        scale_out_file = os.path.join(scales_out_dir, subject, 'body_scales.npy')
        os.makedirs(os.path.dirname(scale_out_file), exist_ok=True)

        os_model = os.path.join(args.data_dir, 'opensim_data', subject, f'{subject}.osim')
        with open(os_model, 'r') as f:
            data = f.read()

        Bs_data = BeautifulSoup(data, "xml")
        body_scales = np.ones((len(body2idx.keys()), 3))

        for body in body2idx.keys():
            b_name = Bs_data.find('Mesh', {'name': f'{body}_geom_1'})
            if b_name is not None:
                scales = b_name.find('scale_factors')
                body_scales[body2idx[body]] = np.fromstring(scales.string, sep=' ')

        os.makedirs(os.path.dirname(scale_out_file), exist_ok=True)
        os.system(f'cp {os_model} {os.path.dirname(scale_out_file)}')
        
        base_os_model = os.path.join(args.data_dir, 'opensim_data', 'generic.osim')
        with open(base_os_model, 'r') as f:
            data = f.read()

        Bs_data = BeautifulSoup(data, "xml")
        base_body_scales = np.ones((len(body2idx.keys()), 3))

        for body in body2idx.keys():
            b_name = Bs_data.find('Mesh', {'name': f'{body}_geom_1'})
            if b_name is not None:
                scales = b_name.find('scale_factors')
                base_body_scales[body2idx[body]] = np.fromstring(scales.string, sep=' ')

        body_scales = body_scales / base_body_scales
        np.save(scale_out_file, body_scales)

        # ---------------------
        #  Get Coordinates
        # ---------------------
        mot_file = glob.glob(os.path.join(args.data_dir, 'opensim_data', subject, f'*/{subject}_{action}_rad.mot'))[0]
        
        out_clip_dir = os.path.join(coords_out_dir, f'{subject}_{action}_{bmi_idx}')
        os.makedirs(out_clip_dir, exist_ok=True)
    
        # rotate osim due to the different definition of XYZ in OpenSim and Blender
        coords, coord_keys = read_mot(mot_file, coord2idx)
        pelvis_pos = coords[:, 3:6]
        rotOsim2Smpl= R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
        coords[:, 3:6] = rotOsim2Smpl.apply(pelvis_pos)
        
        for f in range(coords.shape[0]):
            coords[f, :3] = (rotOsim2Smpl * R.from_euler('ZXY', coords[f, :3])).as_euler('ZXY')

        # subsample frames
        coords = coords[sampled_idx]
        
        coords[:, 10] = coords[:, 9]
        coords[:, 18] = coords[:, 17]
        
        motname = os.path.splitext(str.split(mot_file, '/')[-1])[0]
        out_mot_file = os.path.join(out_clip_dir, motname + '_rotated.mot')
        wrtieCoords2Mot(out_mot_file, coords, coord2idx, True, fps = fps_out)

        for i in range(coords.shape[0]):
            coord_out_file = os.path.join(out_clip_dir, f'frame_{i+first_frame_idx}_coords.npy')
            with open(coord_out_file, 'wb') as f:
                np.save(f, coords[i])

        # ---------------------
        #  Get Points
        # ---------------------
        osim_model_in = np.load(OSIM_MODEL_PATH, allow_pickle=True).item()

        # ------- pytorch version -----------
        joints_info = osim_model_in['joints_info']
        coordinates = osim_model_in['coordinates']
        coord2idx = osim_model_in['coord2idx']
        body2idx = osim_model_in['body2idx']
        coords_range = osim_model_in['coords_range']

        body_info = osim_model_in['body_info']
        ground_info = osim_model_in['ground_info']

        osim_model = OSIM(joints_info, body_info, ground_info, coordinates, coords_range, coord2idx, body2idx)

        coords_arr_all = torch.from_numpy(coords).float().unsqueeze(-1).to(device)
        scales_new_all = torch.from_numpy(body_scales).float().unsqueeze(0).repeat(coords_arr_all.shape[0], 1, 1).to(device)
         
        # m to cm 
        joint_pos_all = osim_model.forward(coords_arr_all, scales_new_all)
        out_clip_dir = os.path.join(points_out_dir, f'{subject}_{action}_{bmi_idx}')
        os.makedirs(out_clip_dir, exist_ok=True)
        
        # subsample
        for i in range(joint_pos_all.shape[0]):
            point_out_file = os.path.join(out_clip_dir, f'frame_{i+first_frame_idx}_joint_points.npy')
            with open(point_out_file, 'wb') as f:
                np.save(f, joint_pos_all[i].cpu().numpy())
