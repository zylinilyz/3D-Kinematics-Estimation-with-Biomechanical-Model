import os
import sys
current_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_pwd}/../src/')
OSIM_MODEL_PATH = f'{current_pwd}/../src/utils/opensim_models/osim_model_scale1.npy'

import glob
import numpy as np
from utils.OSIM import *
import torch

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

def wrtieCoords2Mot(filename, input_coords, coord2idx, isRadian, fps=60):
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

if __name__ == "__main__":
    
    
    for dataset in ['ODAH']:
        
        fps = 60
        home_dir = f'{current_pwd}/../test_outputs/'
        gt_mot_dir = os.path.join(f'{current_pwd}/../data/test_data', dataset, 'ik_groundtruth_sync_final')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for exp_name in ['spatial', 'spatialtemporal']:
            
            print(f'========== DATASET: {dataset}, EXPERIMENT: {exp_name} ========== ')
            
            mot_src_dir = os.path.join(home_dir, dataset, f'ik_mot_files_{exp_name}')
            scale_src_dir = os.path.join(home_dir, dataset, f'osim_model_{exp_name}')
            out_dir = os.path.join(home_dir, dataset, f'final_out_{exp_name}')
            os.makedirs(out_dir, exist_ok=True)
            
            action_dir_list = glob.glob(os.path.join(mot_src_dir, f'*/*'))
            chunk_length = 64
            for action_dir in action_dir_list:
                if os.path.isfile(action_dir):
                    continue
                
                action = str.split(action_dir, '/')[-1]
                subject = str.split(action_dir, '/')[-2]
                
                print(f'===== {subject} ====== {action} ======')
                
                subject_dir =  os.path.join(mot_src_dir, subject)
                
                gt_coords_file = os.path.join(gt_mot_dir, f'{subject}', f'ik_{action}_radians.mot')
                gt_coords, gt_header = read_mot(gt_coords_file, coord2idx)
                n_chunks = int(np.ceil(gt_coords.shape[0]/chunk_length))
                
                mot_files = glob.glob(os.path.join(action_dir, 'ik_*.mot'))
                coords_all = []
                empty_idx = []
                for mot_file_idx in range(n_chunks):
                    
                    mot_file = os.path.join(action_dir, f'ik_{mot_file_idx}_radians.mot')
                    
                    if os.path.exists(mot_file) is False:
                        print('-------------------')
                        print(f'--{mot_file_idx}--')
                        print('-------------------')
                        start_frame = mot_file_idx * chunk_length
                        end_frame = start_frame + chunk_length
                        
                        empty_idx += [[start_frame, end_frame]]
                        coords = np.zeros((chunk_length, len(coord2idx.keys())))
                    else:
                        coords, header = read_mot(mot_file, coord2idx)
                    coords_all.append(coords)
                    
                coords_all = np.concatenate(coords_all, axis=0)
                
                # interpolate
                for interval in empty_idx:
                    start_frame = interval[0]
                    end_frame = interval[1]
                    
                    src_frame_b = start_frame-1
                    src_frame_a = end_frame
                    
                    for f_idx in range(start_frame, end_frame):
                        if src_frame_b < 0:
                            coords_all[f_idx] = coords_all[src_frame_a]
                        elif src_frame_a > coords_all.shape[0]-1:
                             coords_all[f_idx] = coords_all[src_frame_b]
                        else:
                            w_b = (src_frame_a - f_idx) / (src_frame_a - src_frame_b)
                            w_a = (f_idx - src_frame_b) / (src_frame_a - src_frame_b)
                            coords_all[f_idx] = w_a * coords_all[src_frame_a] + w_b * coords_all[src_frame_b]
                        
                if os.path.exists(os.path.join(gt_mot_dir, f'{subject}', f'ik_{action}_radians.mot')) is False:
                    continue
                
                n_frames = min(coords_all.shape[0], gt_coords.shape[0])
                coords_all = coords_all[:n_frames]
                
                # save coords
                out_filename = os.path.join(out_dir, subject, f'ik_{action}_radians.mot')
                os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                wrtieCoords2Mot(out_filename, coords_all, coord2idx, True, fps=fps)
                
                scale_files = glob.glob(os.path.join(scale_src_dir, subject, action, '*_scaled.npy'))
                scale_all = []
                empty_idx = []
                for scale_file_idx in range(n_chunks):
                    
                    scale_file = os.path.join(scale_src_dir, subject, action, f'osim_scales_{scale_file_idx}_scaled.npy')
                    
                    if os.path.exists(scale_file) is False:
                        print('-------------------')
                        print(f'--{scale_file}--')
                        print('-------------------')
                        scale = np.zeros((chunk_length, len(body2idx.keys()), 3))
                        start_frame = scale_file_idx * chunk_length
                        end_frame = start_frame + chunk_length
                        empty_idx += [list(range(start_frame, end_frame))]
                    else:
                        scale = np.load(scale_file)
                        
                    if len(scale.shape) < 3:
                        scale = scale[None]
                    #print("------", scale.shape)
                    #scale = np.repeat(scale, 64, axis=0)
                    scale_all.append(scale)
                    
                scale_all = np.concatenate(scale_all, axis=0)
                
                # interpolate
                for interval in empty_idx:
                    start_frame = interval[0]
                    src_frame = start_frame-1
                    
                    if src_frame < 0:
                        src_frame = interval[-1]
                        scale_all[interval] = coords_all[src_frame]
                    else:
                        scale_all[interval] = scale_all[src_frame][None]
                    
                scale_all = scale_all[:n_frames]
                # save scales
                out_filename = os.path.join(out_dir, subject, f'osim_scales_{action}_scaled.npy')
                os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                np.save(out_filename, scale_all)
                
                
                #######
                # resample
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

                coords_arr_all = torch.from_numpy(coords_all).float().unsqueeze(-1).to(device)
                scales_new_all = torch.from_numpy(scale_all).float().to(device)
                
                # m to cm 
                joint_pos_all, joint_rot_all = osim_model.forward(coords_arr_all, scales_new_all, isTrain=False)
                
                joint_info_filename = os.path.join(out_dir, subject, f'{subject}_{action}_joint.npy')
                body_info_filename = os.path.join(out_dir, subject, f'{subject}_{action}_body.npz')
                os.makedirs(os.path.dirname(joint_info_filename), exist_ok=True)
                
                joint_pos_all = joint_pos_all.cpu().numpy()
                joint_rot_all = joint_rot_all.cpu().numpy()
                
                joint_info_all = []
                for idx in range(joint_pos_all.shape[0]):
                    joint_info = {}
                    for pb_name in body2idx.keys():
                        joint_info[pb_name] = {}
                        joint_info[pb_name]['pos'] = joint_pos_all[idx, 22+body2idx[pb_name]]
                        joint_info[pb_name]['rot'] = joint_rot_all[idx, 22+body2idx[pb_name]]
                    joint_info_all.append(joint_info)
                    
                body_pos_all = []
                body_rot_all = []
                for idx in range(joint_pos_all.shape[0]):
                    body_pos_all.append(joint_pos_all[idx, :22])
                    body_rot_all.append(joint_rot_all[idx, :22])
                    
                np.save(joint_info_filename, joint_info_all)
                np.savez(body_info_filename, bodyPos=body_pos_all, bodyRot=body_rot_all)
                
                