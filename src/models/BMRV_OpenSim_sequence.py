import torch
from collections import OrderedDict
from utils.OSIM import  *
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np
import copy

from einops import rearrange
from utils.util_rotation import *
from utils.util_projection import *
#SEG_SIZE = 32

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

ANGLES = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
#RANGE_ANGLES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 34, 35]
# exclude pelvis trans, rot, arm rot
RANGE_ANGLES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31, 35, 36, 37, 38]
PELVIS_ROT = [0, 1, 2]
PELVIS_POS = [3, 4, 5]
ARM_ROT_R = [25, 26, 27]
ARM_ROT_L = [32, 33, 34]
ARM_ROT = [25, 26, 27, 32, 33, 34]

FREE_ROT = [0, 1, 2, 25, 26, 27, 32, 33, 34]
NORM_ANGLES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31, 35, 36, 37, 38]

PELVIS_ROOT_IDX = 22

class L1_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t, mean=True):
        if mean is True:
            return (torch.abs(y_t - y_prime_t)).mean()
        else:
            return torch.abs(y_t - y_prime_t)
    
class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            # first round training, copy lvd weights
            if self._opt.load_epoch <= 0:
                
                if self._opt.use_pretrained == 1:
                    self._load_lvd()
                    # copy parameters
                    self._img_encoder.image_filter = copy.deepcopy(self._img_encoder_lvd.image_filter)
                    del self._img_encoder_lvd
                    torch.cuda.empty_cache()
                
                elif self._opt.use_pretrained == 2:
                
                    self._load_frame_encoder()
                    self._img_encoder.load_state_dict(self._frame_encoder.state_dict(), strict=False)
                
                    del self._frame_encoder
                    torch.cuda.empty_cache()
                
                if torch.cuda.is_available():
                    self._img_encoder.to(self.device)

            self._init_train_vars()
            
            if self._opt.load_epoch > 0:
                self.load()
        else:
            load_path = self._opt.pretrained_checkpoint_path
            assert os.path.exists(
                load_path), 'Weights file not found. Have you trained a model!? We are not providing one  {}'.format(load_path)

            self._img_encoder.load_state_dict(torch.load(load_path, map_location='cpu'))
            print('loaded net: %s' % load_path)
            

        # fix feature extractor's weights
        if self._opt.free_hg == 0:
            for p in self._img_encoder.image_filter.parameters():
                p.requires_grad = False

        # init
        self._init_losses()
        osim_model_in = np.load(self._opt.home_dir + 'utils/opensim_models/osim_model_scale1.npy', allow_pickle=True).item()

        joints_info = osim_model_in['joints_info']
        coordinates = osim_model_in['coordinates']
        coord2idx = osim_model_in['coord2idx']
        body2idx = osim_model_in['body2idx']

        ground_info = osim_model_in['ground_info']
        body_info = osim_model_in['body_info']
        coords_range = osim_model_in['coords_range']
    
        self.osim_model = OSIM(joints_info, body_info, ground_info, coordinates, coords_range, coord2idx, body2idx).to(self.device)
        self.angle_range = self.osim_model.coords_range
        

        self.pred_coords = None
        self.pred_scales = None
        self.gt_coords = None
        self.gt_scales = None
        self.inp_images = None
        self._reconstruction_error = torch.FloatTensor([0]).cpu().data.numpy()

        self.sp_weight = 0.5
        self.tp_weight = 0.5
        
        self.basic_loss = L1_Loss()
  
    def _load_lvd(self):
        # load pre-trained weight
        self._img_encoder_lvd = NetworksFactory.get_by_name('LVD_images', num_view = 1, pred_dimensions=6890*3)

        load_path = self._opt.pretrained_checkpoint_path
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one {}'.format(load_path)

        self._img_encoder_lvd.load_state_dict(torch.load(load_path, map_location='cpu'))
        print('loaded net: %s' % load_path)

    def _load_frame_encoder(self):
        # load pre-trained weight
        self._frame_encoder = NetworksFactory.get_by_name('BMRV_image_network'
                                            , num_view = 2 #self._opt.num_view
                                            , num_input_points = self._opt.num_points
                                            , reduced_dims = self._opt.reduced_dim
                                            , pred_dims = 36+22*3
                                            , input_point_dimensions = 3
                                            , hg_heads = self._opt.hg_heads
                                            , hg_out = self._opt.hg_out)

        load_path = self._opt.pretrained_checkpoint_path
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one {}'.format(load_path)

        self._frame_encoder.load_state_dict(torch.load(load_path, map_location='cpu'))
        print('loaded net: %s' % load_path)
        
    def _init_create_networks(self):
        # generator network
        self._img_encoder = self._create_img_encoder()
        self._img_encoder.init_weights()
        if torch.cuda.is_available():
            #self._img_encoder = nn.DataParallel(self._img_encoder)
            self._img_encoder.to(self.device)

    def _create_img_encoder(self):
        return NetworksFactory.get_by_name('BMRV_sequence_network'
                                            , num_view = self._opt.num_view
                                            , num_input_points = self._opt.num_points
                                            , reduced_dims = self._opt.reduced_dim
                                            , seq_length = self._opt.chunk_length
                                            , frame_pred_dims = 36+22*3
                                            , seq_pred_dims = 36+22*3
                                            , input_point_dimensions = 3
                                            , hg_heads = self._opt.hg_heads
                                            , hg_out = self._opt.hg_out)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_G_temporal = self._opt.lr_G_temporal

        if self._opt.optimizer == 'AdamW':
            self._optimizer_img_encoder = torch.optim.AdamW([ {'name': 'image_filter', 'params': self._img_encoder.image_filter.parameters()},
                                                            {'name': 'point2osim', 'params': self._img_encoder.point2osim.parameters()},
                                                            {'name': 'frame2seq', 'params': self._img_encoder.frame2seq.parameters(), 'lr': self._current_lr_G_temporal}],
                                                            lr=self._current_lr_G, weight_decay=0.0001)
        else:
            self._optimizer_img_encoder = torch.optim.Adam([ {'name': 'image_filter', 'params': self._img_encoder.image_filter.parameters()},
                                                         {'name': 'point2osim', 'params': self._img_encoder.point2osim.parameters()},
                                                         {'name': 'frame2seq', 'params': self._img_encoder.frame2seq.parameters(), 'lr': self._current_lr_G_temporal}]
                                                     , lr=self._current_lr_G, betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        
            
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer_img_encoder, self._opt.nepochs_decay, eta_min=0.0000001, last_epoch=-1, verbose=True)
            
        self.mesh = None

    def set_epoch(self, epoch):
        self.iepoch = epoch

    def _init_losses(self):
        # init losses G
        self._sp_loss_all = torch.FloatTensor([0]).to(self.device)
        self._sp_loss_angles = torch.FloatTensor([0]).to(self.device)
        self._sp_loss_pelvis_pos = torch.FloatTensor([0]).to(self.device)
        self._sp_loss_points = torch.FloatTensor([0]).to(self.device)
        self._sp_loss_points_proj = torch.FloatTensor([0]).to(self.device)
        self._sp_loss_scales = torch.FloatTensor([0]).to(self.device)
        self._sp_loss_angle_range = torch.FloatTensor([0]).to(self.device)
        self._sp_reg_scales = torch.FloatTensor([0]).to(self.device)
        self._sp_loss_vel = torch.FloatTensor([0]).to(self.device)
        self._sp_loss_acc = torch.FloatTensor([0]).to(self.device)

        self._tp_loss_all = torch.FloatTensor([0]).to(self.device)
        self._tp_loss_angles = torch.FloatTensor([0]).to(self.device)
        self._tp_loss_pelvis_pos = torch.FloatTensor([0]).to(self.device)
        self._tp_loss_points = torch.FloatTensor([0]).to(self.device)
        self._tp_loss_points_proj = torch.FloatTensor([0]).to(self.device)
        self._tp_loss_scales = torch.FloatTensor([0]).to(self.device)
        self._tp_loss_angle_range = torch.FloatTensor([0]).to(self.device)
        self._tp_reg_scales = torch.FloatTensor([0]).to(self.device)
        self._tp_loss_vel = torch.FloatTensor([0]).to(self.device)
        self._tp_loss_acc = torch.FloatTensor([0]).to(self.device)

        self._loss_all = torch.FloatTensor([0]).to(self.device)

    def dist_angle(self, angle_1, angle_2):
        d = (
            self.basic_loss(torch.cos(angle_1), torch.cos(angle_2), mean=False) +
            self.basic_loss(torch.sin(angle_1), torch.sin(angle_2), mean=False)
        )
        return d
            
    def angle_interval_loss(self, val, lims):
        # val: (B, 39)
        # lims: (39, 2)
        min_v = lims[:, 0].unsqueeze(0).unsqueeze(-1)
        max_v = lims[:, 1].unsqueeze(0).unsqueeze(-1)
        d1 = self.basic_loss(val[:, RANGE_ANGLES], min_v[:, RANGE_ANGLES], mean=False)
        d2 = self.basic_loss(val[:, RANGE_ANGLES], max_v[:, RANGE_ANGLES], mean=False)
        mask = (val[:, RANGE_ANGLES] >= min_v[:, RANGE_ANGLES]).float() * (val[:, RANGE_ANGLES] <= max_v[:, RANGE_ANGLES]).float()
        # remove tx, ty, tz
        loss_interval = mask * 0 + (1 - mask) * torch.min(d1, d2)

        return loss_interval

    def get_rot_loss(self, pred_angle, target_angle):
        angle_dist = self.basic_loss(torch.cos(pred_angle), torch.cos(target_angle), mean=False) + \
                    self.basic_loss(torch.sin(pred_angle), torch.sin(target_angle), mean=False)
            
        return angle_dist


    def scale_interval_loss(self, val):
        min_v = 0.5
        max_v = 2
        d1 = torch.abs(val - min_v)
        d2 = torch.abs(val - max_v)
        mask = (val >= min_v).float() * (val <= max_v).float()
        # remove tx, ty, tz
        loss_interval = mask * 0 + (1 - mask) * torch.min(d1, d2)

        return loss_interval
    
    def scale_smoothness(self, pred_scales):
        # pred_scales (B, f, 22, 3)
        b, f, p, c = pred_scales.shape
        scales = rearrange(pred_scales, 'b f p c -> (b f) p c')
        scales_overall = torch.prod(scales, dim=-1, keepdim=True)
        scales_body_var = torch.var(scales_overall, dim=1).mean()
        scales_dim_var = torch.var(scales, dim=-1).mean()
        
        return scales_dim_var + scales_body_var

    def set_input(self, input):
        input_image = input['input_image'].float()
        self._input_points = input['input_points'].float().to(self.device)
        
        self._input_points = rearrange(self._input_points, 'b f v p c -> (b f) v p c')
        self._input_points_full = input['input_points_full'].float().to(self.device)
        self._input_points_full = rearrange(self._input_points_full, 'b f p c -> (b f) p c')
        self._target_scales = input['target_scales'].float().to(self.device)
        self._target_scales = self._target_scales.unsqueeze(1).repeat(1, self._opt.chunk_length, 1, 1)
        
        self._target_points = input['target_points'].float().to(self.device)
        self._target_coords = input['target_coords'].float().to(self.device)
        self._input_cam_proj_inv = input['input_cam_proj_inv'].float().to(self.device)
            
        self._input_meta = input['input_meta'].float().to(self.device)
        self._input_meta = rearrange(self._input_meta, 'b f v c -> (b f) v c')
        self._input_cam_proj = input['input_cam_proj'].float().to(self.device)
        n_ch = input_image.shape[2] 
        seg_size = n_ch // self._opt.num_view
        image_ = input_image[:, :, :seg_size-1, :, :]
        for i in range(1, self._opt.num_view):
            image_ = torch.cat((image_, input_image[:, :, seg_size*i:seg_size*(i+1)-1, :, :]), dim=2)

        self._input_image = image_
        self._input_image = rearrange(self._input_image, 'b f c h w -> (b f) c h w')

        return

    def set_train(self):
        self._img_encoder.train()
        self._is_train = True

    def set_eval(self):
        self._img_encoder.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, interpolate=0, resolution=128):
        if not self._is_train:
            # Reconstruct first 
            with torch.no_grad():
                
                # frame prediction
                f = self._opt.chunk_length
                seg_size = self._opt.seg_size
                iter = int(self._input_image.shape[0] / seg_size)
                pred_feats = []
                for i in range(iter):
                    img = self._input_image[i*seg_size:(i+1)*seg_size].to(self.device)
                    points = self._input_points[i*seg_size:(i+1)*seg_size].to(self.device)
                    points_full = self._input_points_full[i*seg_size:(i+1)*seg_size].to(self.device)
                        
                    self._img_encoder(img)
                    pred_tmp = self._img_encoder.query_frame(points, points_full)
                    pred_feats.append(pred_tmp)

                    # clean cache
                    del img, points, points_full
                    torch.cuda.empty_cache()

                pred_feats = torch.cat(pred_feats, dim=0)
                pred_feats = rearrange(pred_feats, '(b f) p c -> b f p c', f = f)
                
                self.inp_images = self._input_image[:f].data.numpy()
                
                #tgt_idx = np.arange(f)
                if self._opt.temporal_only == 0:
                    pred_coords, pred_scales = self.reform_prediction(pred_feats)
                    self._sp_loss_all = self.get_spatial_loss(pred_coords, pred_scales)
            
                pred = self._img_encoder.query_sequence(pred_feats)
                pred_coords, pred_scales = self.reform_prediction(pred)
                self._tp_loss_all, pred_points = self.get_temporal_loss(pred_coords, pred_scales)
                
                if self._opt.temporal_only == 0:
                    self._loss_all = self.sp_weight * self._sp_loss_all + self.tp_weight * self._tp_loss_all
                else:
                    self._loss_all = self._tp_loss_all
                
                self.pred_coords = pred_coords[0].cpu().data.numpy()
                self.pred_scales = pred_scales[0].cpu().data.numpy()
                self.target_coords = self._target_coords[:, :, :, :][0].cpu().data.numpy()
                self.target_scales = self._target_scales[:, :, :, :][0].cpu().data.numpy()
                self._reconstruction_error = self.get_MPJPE(pred_points, self._target_points[:, :, :, :]).cpu().data.numpy()
                # clean cache
                del pred, pred_feats, pred_coords, pred_scales, pred_points
                del self._target_scales, self._target_points
                self._img_encoder.clean_feat()
                torch.cuda.empty_cache()
        return
        
    def optimize_parameters(self):
        if self._is_train:
        
            self._optimizer_img_encoder.zero_grad(set_to_none=True)
            
            # frame prediction
            f = self._opt.chunk_length

            seg_size = self._opt.seg_size
            iter = int(self._input_image.shape[0] / seg_size)
            pred_feats = []
            for i in range(iter):
                img = self._input_image[i*seg_size:(i+1)*seg_size].to(self.device)
                points = self._input_points[i*seg_size:(i+1)*seg_size].to(self.device)
                points_full = self._input_points_full[i*seg_size:(i+1)*seg_size].to(self.device)
                    
                self._img_encoder(img)
                pred_tmp = self._img_encoder.query_frame(points, points_full)
                pred_feats.append(pred_tmp)

                # clean cache
                del img, points, points_full
                torch.cuda.empty_cache()
    
            pred_feats = torch.cat(pred_feats, dim=0)
            pred_feats = rearrange(pred_feats, '(b f) p c -> b f p c', f = f)
            
            #tgt_idx = np.arange(f)
            
            if self._opt.temporal_only == 0:
                pred_coords, pred_scales = self.reform_prediction(pred_feats)
                self._sp_loss_all = self.get_spatial_loss(pred_coords, pred_scales)
            
            # temporal prediction
            pred = self._img_encoder.query_sequence(pred_feats)
            
            pred_coords, pred_scales = self.reform_prediction(pred)
            self._tp_loss_all, pred_points = self.get_temporal_loss(pred_coords, pred_scales)
            
            if self._opt.temporal_only == 0:
                self._loss_all = self.sp_weight * self._sp_loss_all + self.tp_weight * self._tp_loss_all
            else:
                self._loss_all = self._tp_loss_all
            
            self._loss_all.backward()
            #########################################################
            self._optimizer_img_encoder.step()
            
            # clean cache
            del pred, pred_feats, pred_coords, pred_scales, pred_points
            del self._target_scales, self._target_points
            self._img_encoder.clean_feat()
            torch.cuda.empty_cache()

    def reform_prediction(self, pred):
        _B = pred.shape[0]
        _F = pred.shape[1]

        pred_angles = pred[:, :, :36]
        pred_pelvis_pos =  torch.zeros((_B, _F, 3, 1)).float().to(self.device)
        pred_coords = torch.concatenate((pred_angles[:, :, :3], pred_pelvis_pos, pred_angles[:, :, 3:]), dim=2)
        pred_scales = pred[:, :, 36:].reshape(_B, _F, -1, 3) / 8 + 1.0
        
        return pred_coords, pred_scales

    def get_spatial_loss(self, pred_coords, pred_scales):
        
        self._sp_loss_angles, self._sp_loss_angle_range, self._sp_loss_pelvis_pos, self._sp_loss_scales, self._sp_reg_scales = self.get_param_loss(pred_coords, pred_scales)
        
        self._sp_loss_points, pred_points = self.get_point_loss(pred_coords, pred_scales)
                
        loss_list = [self._sp_loss_angles, self._sp_loss_scales]
         
        # point_loss
        loss_list += [self._sp_loss_points]
        
        # range_loss
        loss_list += [self._sp_loss_angle_range]
        
        loss = torch.stack(loss_list).sum()
        
        return loss

    def get_temporal_loss(self, pred_coords, pred_scales):
        
        self._tp_loss_angles, self._tp_loss_angle_range, self._tp_loss_pelvis_pos, self._tp_loss_scales, self._tp_reg_scales = self.get_param_loss(pred_coords, pred_scales)
        
        self._tp_loss_points, pred_points = self.get_point_loss(pred_coords, pred_scales)
            
        loss_list = [self._tp_loss_angles, self._tp_loss_scales]
        
        # point_loss
        loss_list += [self._tp_loss_points]
            
        # range_loss
        loss_list += [self._tp_loss_angle_range]
            
        loss = torch.stack(loss_list).sum()
        
        return loss, pred_points

    def get_MPJPE_L1(self, pred_points, target_points):
        
        pred3d = pred_points - pred_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        gt3d = target_points - target_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        
        return self.basic_loss(gt3d, pred3d) * self._opt.point_weight
    
    def get_MPJPE_L1_proj(self, pred_points, target_points):
        # project to 2D
        _B = pred_points.shape[0]
        _F = pred_points.shape[1]
        pred_points = rearrange(pred_points, 'b f p c -> (b f) p c')
        target_points = rearrange(target_points, 'b f p c -> (b f) p c')
        
        point_all = torch.concatenate((pred_points, target_points), dim=0) * 1000.0
        point_all = torch.concatenate([-point_all[:,:,2:], -point_all[:,:,0:1], point_all[:,:,1:2]], dim=1)
        
        point_2d_front = proj_3d_to_2d(point_all, self._input_cam_proj[0, 0, :, :], self._input_meta[:, 0, :].repeat(2, 1))
        point_2d_side = proj_3d_to_2d(point_all, self._input_cam_proj[0, 1, :, :], self._input_meta[:, 1, :].repeat(2, 1))
        
    
        pred2d = torch.concatenate((point_2d_front[:_B*_F], point_2d_side[:_B*_F]), dim=0)
        gt2d = torch.concatenate((point_2d_front[_B*_F:], point_2d_side[_B*_F:]), dim=0)
        
        pred_points = rearrange(pred_points, '(b f) p c -> b f p c', b=_B)
        target_points = rearrange(target_points, '(b f) p c -> b f p c', b=_B)
        
        pred2d = pred2d - pred2d[:, PELVIS_ROOT_IDX].unsqueeze(1) 
        gt2d = gt2d - gt2d[:, PELVIS_ROOT_IDX].unsqueeze(1) 
        
        return self.basic_loss(gt2d, pred2d) * 0.1 #* self._opt.point_weight
    
    def get_MPJPE(self, pred_points, target_points):
        
        pred3d = pred_points - pred_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        gt3d = target_points - target_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 

        return torch.sqrt(((gt3d - pred3d)**2).sum(-1)).mean()  * 1000
    
    def get_MPJVE_L1(self, pred_points, target_points):
        pred3d = pred_points - pred_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        gt3d = target_points - target_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        
        velocity = pred3d[:, 1:, :, :] - pred3d[:, :-1, :, :]
        velocity_gt = gt3d[:, 1:, :, :] - gt3d[:, :-1, :, :]
        
        return self.basic_loss(velocity, velocity_gt) * self._opt.point_weight
    
    def get_MPJVE(self, pred_points, target_points):
        pred3d = pred_points - pred_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        gt3d = target_points - target_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        
        velocity = pred3d[:, 1:, :, :] - pred3d[:, :-1, :, :]
        velocity_gt = gt3d[:, 1:, :, :] - gt3d[:, :-1, :, :]
        return torch.sqrt(((velocity - velocity_gt)**2).sum(-1)).mean()  * 1000

    def get_MPJAE_L1(self, pred_points, target_points):
        pred3d = pred_points - pred_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        gt3d = target_points - target_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        
        velocity = pred3d[:, 1:, :, :] - pred3d[:, :-1, :, :]
        velocity_gt = gt3d[:, 1:, :, :] - gt3d[:, :-1, :, :]
        
        acceleration = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]
        acceleration_gt = velocity_gt[:, 1:, :, :] - velocity_gt[:, :-1, :, :]

        return self.basic_loss(acceleration, acceleration_gt) * self._opt.point_weight
    
    def get_MPJAE(self, pred_points, target_points):
        pred3d = pred_points - pred_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        gt3d = target_points - target_points[:, :, PELVIS_ROOT_IDX].unsqueeze(2) 
        
        velocity = pred3d[:, 1:, :, :] - pred3d[:, :-1, :, :]
        velocity_gt = gt3d[:, 1:, :, :] - gt3d[:, :-1, :, :]
        
        acceleration = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]
        acceleration_gt = velocity_gt[:, 1:, :, :] - velocity_gt[:, :-1, :, :]

        return torch.sqrt(((acceleration - acceleration_gt)**2).sum(-1)).mean()  * 1000

    def get_l2norm_acc(self, pred):
        velocity = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        acceleration = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]
        
        return ((acceleration * self._opt.point_weight)**2).mean()

    def get_param_loss(self, pred_coords, pred_scales):
        
        loss_free = self.get_rot_loss(pred_coords[:, :, FREE_ROT, :], self._target_coords[:, :, FREE_ROT, :]).mean()
        loss_dist =  self.basic_loss(pred_coords[:, :, NORM_ANGLES, :], self._target_coords[:, :, NORM_ANGLES, :])
        loss_angles = loss_free + loss_dist
        
        loss_pelvis_pos = torch.tensor(0).float().to(self.device)
        
        pred_coords = rearrange(pred_coords, 'b f p c -> (b f) p c')
        loss_angle_range = self.angle_interval_loss(pred_coords, self.angle_range).mean()
        
        loss_scales = self.basic_loss(pred_scales, self._target_scales[:, :, :, :])
        # pred_scales (B, f, 22, 3)
        reg_scales = self.scale_smoothness(pred_scales)

        return loss_angles, loss_angle_range, loss_pelvis_pos, loss_scales, reg_scales

    def get_point_loss(self, pred_coords, pred_scales):
        _F = pred_coords.shape[1]
        pred_coords = rearrange(pred_coords, 'b f p c -> (b f) p c')
        pred_scales = rearrange(pred_scales, 'b f p c -> (b f) p c')
        pred_points = self.osim_model.forward(pred_coords, pred_scales)
        pred_points = rearrange(pred_points, '(b f) p c -> b f p c', f = _F)
        
        loss_points = self.get_MPJPE_L1(pred_points, self._target_points[:, :, :, :])
        
        return loss_points, pred_points


    def get_current_errors(self):
        loss_dict = OrderedDict([
                                 ('Angle_loss_sp', self._sp_loss_angles.cpu().data.numpy()),
                                 ('Angle_loss_tp', self._tp_loss_angles.cpu().data.numpy()),
                                 ('PelvisPos_loss_sp', self._sp_loss_pelvis_pos.cpu().data.numpy()),
                                 ('PelvisPos_loss_tp', self._tp_loss_pelvis_pos.cpu().data.numpy()),
                                 ('AngleRange_loss_sp', self._sp_loss_angle_range.cpu().data.numpy()),
                                 ('AngleRange_loss_tp', self._tp_loss_angle_range.cpu().data.numpy()),
                                 ('Scale_loss_sp', self._sp_loss_scales.cpu().data.numpy()),
                                 ('Scale_loss_tp', self._tp_loss_scales.cpu().data.numpy()),
                                 ('Distance_loss_sp', self._sp_loss_points.cpu().data.numpy()),
                                 ('Distance_loss_tp', self._tp_loss_points.cpu().data.numpy()),
                                 ('Distance_loss_sp_proj', self._sp_loss_points_proj.cpu().data.numpy()),
                                 ('Distance_loss_tp_proj', self._tp_loss_points_proj.cpu().data.numpy()),
                                 ('Total_loss_sp', self._sp_loss_all.cpu().data.numpy()),
                                 ('Total_loss_tp', self._tp_loss_all.cpu().data.numpy()),
                                 ('Total_loss', self._loss_all.cpu().data.numpy()),
                                 ('Val. OpenSim reconstruction error', self._reconstruction_error),
                                ])
        return loss_dict

    def get_current_scalars(self):
            
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_G_temporal', self._current_lr_G_temporal)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        if type(self.pred_coords) == type(None):
            return visuals

        visuals['input_image'] = []
        n_frames = self._opt.chunk_length
        seg_size = self.inp_images.shape[1] // self._opt.num_view
        for j in range(n_frames):
            for i in range(self._opt.num_view):
                view = (self.inp_images[j, i * seg_size: (i+1) * seg_size].transpose(1,2,0)* [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
                visuals['input_image'].append(view)
   
        visuals['osim_prediction'] = self.pred_coords
        visuals['osim_groundtruth'] = self.gt_coords

        self.pred_coords = None
        self.pred_scales = None
        self.inp_images = None
        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._img_encoder, 'img_encoder', label)

        # save optimizers
        self._save_optimizer(self._optimizer_img_encoder, 'img_encoder', label)
        
        # save lr cheduler
        self._save_lr_scheduler(self._lr_scheduler, 'img_encoder', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._img_encoder, 'img_encoder', load_epoch)

        if self._is_train:
            # load optimizers
            self._load_optimizer(self._optimizer_img_encoder, 'img_encoder', load_epoch)
            
            # load lr cheduler
            self._load_lr_scheduler(self._lr_scheduler, 'img_encoder', load_epoch)


    def update_learning_rate(self):
        self._lr_scheduler.step()
        for param_group in self._optimizer_img_encoder.param_groups:
            if param_group['name'] == 'frame2seq':
                self._current_lr_G_temporal = param_group['lr']
                break
            else:
                self._current_lr_G = param_group['lr']
                break
        