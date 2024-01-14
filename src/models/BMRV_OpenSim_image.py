import torch
from collections import OrderedDict
from utils.OSIM import  *
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np
from torch import nn
import copy
from utils.util_rotation import *
from utils.util_projection import *

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

class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        # B, N, 3, 3
        #m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        m = torch.matmul(m1, m2.transpose(2,3))
        
        cos = (  m[:,:, 0,0] + m[:,:, 1,1] + m[:,:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
         
        return torch.mean(theta)
    
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_t, y_prime_t, mean=True):
        ey_t = y_t - y_prime_t
        if mean is True:
            return (torch.log(torch.cosh(ey_t + 1e-12))).mean()
        else:
            return torch.log(torch.cosh(ey_t + 1e-12))
            
    
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
        self._name = 'model1'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            # first round training, copy lvd weights
            if self._opt.num_view != 1 and self._opt.load_epoch <= 0:
                self._load_lvd()
                # copy parameters
                self._img_encoder.image_filter = copy.deepcopy(self._img_encoder_lvd.image_filter)
            
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
        #self.SMPL = SMPL('utils/neutral_smpl_with_cocoplus_reg.txt', obj_saveable = True).to(self.device)
        
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
        self._reconstruction_error = torch.FloatTensor([0]).cpu().data.numpy()
        
        self.basic_loss = L1_Loss()
            
  
    def _load_lvd(self):
        # load pre-trained weight
        self._img_encoder_lvd = NetworksFactory.get_by_name('LVD_images', num_view = 1, pred_dimensions=6890*3)

        load_path = self._opt.pretrained_checkpoint_path
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one {}'.format(load_path)

        self._img_encoder_lvd.load_state_dict(torch.load(load_path, map_location='cpu'))
        print('loaded net: %s' % load_path)

    
        
    def _init_create_networks(self):
        # generator network
        self._img_encoder = self._create_img_encoder()
        self._img_encoder.init_weights()
        if torch.cuda.is_available():
            self._img_encoder.to(self.device)

    def _create_img_encoder(self):
        return NetworksFactory.get_by_name('BMRV_image_network'
                                            , num_view = self._opt.num_view
                                            , num_input_points = self._opt.num_points
                                            , reduced_dims = self._opt.reduced_dim
                                            , pred_dims = 36+22*3
                                            , input_point_dimensions = 3
                                            , hg_heads = self._opt.hg_heads
                                            , hg_out = self._opt.hg_out)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G

        # initialize optimizers
        self._optimizer_img_encoder = torch.optim.Adam(self._img_encoder.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])

        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer_img_encoder, self._opt.nepochs_decay, eta_min=0.0000001, last_epoch=-1, verbose=True)
            
        self.mesh = None

    def set_epoch(self, epoch):
        self.iepoch = epoch

    def _init_losses(self):
        # init losses G
        self._loss_all = torch.FloatTensor([0]).to(self.device)
        self._loss_angles = torch.FloatTensor([0]).to(self.device)
        self._loss_pelvis_pos = torch.FloatTensor([0]).to(self.device)
        self._loss_points = torch.FloatTensor([0]).to(self.device)
        self._loss_points_proj = torch.FloatTensor([0]).to(self.device)
        self._loss_scales = torch.FloatTensor([0]).to(self.device)
        self._loss_angle_range = torch.FloatTensor([0]).to(self.device)

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

    def set_input(self, input):
        input_image = input['input_image'].float()
        
        self._target_coords = input['target_coords'].float().to(self.device)
        
        self._input_points = input['input_points'].float().to(self.device)
        self._input_points_full = input['input_points_full'].float().to(self.device)
        
        self._target_points = input['target_points'].float().to(self.device)
        self._target_scales = input['target_scales'].float().to(self.device)
        self._input_cam_proj_inv = input['input_cam_proj_inv'].float().to(self.device)
        
        # input_meta: B, V, ... -> (B, V, ...)
        self._input_meta = input['input_meta'].float().to(self.device)
        
        # _input_cam_proj: B, V, ...
        self._input_cam_proj = input['input_cam_proj'].float().to(self.device)
        
        # remove mask (mask is added regardless of the input setting)
        #print('set_input: ', input_image.shape)
        seg_size = input_image.shape[1] // self._opt.num_view
        image_ = input_image[:,:seg_size-1, :, :]
        for i in range(1, self._opt.num_view):
            image_ = torch.cat((image_, input_image[:, seg_size*i:seg_size*(i+1)-1, :, :]), dim=1)

        self._input_image = image_.to(self.device)

        return 

    def set_train(self):
        self._img_encoder.train()
        self._is_train = True

    def set_eval(self):
        self._img_encoder.eval()
        self._is_train = False
    
    def get_MPJPE_L1(self, pred_points, target_points):
        pred3d = pred_points - pred_points[:, PELVIS_ROOT_IDX].unsqueeze(1) 
        gt3d = target_points - target_points[:, PELVIS_ROOT_IDX].unsqueeze(1) 
        
        return self.basic_loss(gt3d, pred3d) * self._opt.point_weight
    
    def get_MPJPE_L1_proj(self, pred_points, target_points):
        # project to 2D
        _B = pred_points.shape[0]
        point_all = torch.concatenate((pred_points, target_points), dim=0) * 1000.0
        point_all = torch.concatenate([-point_all[:,:,2:], -point_all[:,:,0:1], point_all[:,:,1:2]], dim=1)
        
        point_2d_front = proj_3d_to_2d(point_all, self._input_cam_proj[0, 0, :, :], self._input_meta[:, 0, :].repeat(2, 1))
        point_2d_side = proj_3d_to_2d(point_all, self._input_cam_proj[0, 1, :, :], self._input_meta[:, 1, :].repeat(2, 1))
        
        pred2d = torch.concatenate((point_2d_front[:_B], point_2d_side[:_B]), dim=0)
        gt2d = torch.concatenate((point_2d_front[_B:], point_2d_side[_B:]), dim=0)
        
        pred2d = pred2d - pred2d[:, PELVIS_ROOT_IDX].unsqueeze(1) 
        gt2d = gt2d - gt2d[:, PELVIS_ROOT_IDX].unsqueeze(1) 
        
        return self.basic_loss(gt2d, pred2d) * 0.1#* self._opt.point_weight
    
    def get_MPJPE(self, pred_points, target_points):
        pred3d = pred_points - pred_points[:, PELVIS_ROOT_IDX].unsqueeze(1) 
        gt3d = target_points - target_points[:, PELVIS_ROOT_IDX].unsqueeze(1) 

        return torch.sqrt(((gt3d - pred3d)**2).sum(-1)).mean() * 1000
    
    def forward(self, keep_data_for_visuals=False, interpolate=0, resolution=128):
        if not self._is_train:
            # Reconstruct first 
            _B = self._input_image.shape[0]
            with torch.no_grad():
                self._img_encoder(self._input_image)
                # pred: (B, 39+22*3, 1)
                pred = self._img_encoder.query_frame(self._input_points, self._input_points_full)
                #pred = pred.squeeze(2)
                
                pred_angles = pred[:, :36]
                pred_pelvis_pos = torch.zeros((_B, 3, 1)).float().to(self.device)
                pred_coords = torch.concatenate((pred_angles[:, :3], pred_pelvis_pos, pred_angles[:, 3:]), dim=1)
                pred_scales = pred[:, 36:].reshape(_B, -1, 3) / 8 + 1.0
            
                
                loss_free = self.get_rot_loss(pred_coords[:, FREE_ROT], self._target_coords[:, FREE_ROT]).mean()
                loss_dist = self.basic_loss(pred_coords[:, NORM_ANGLES], self._target_coords[:, NORM_ANGLES])
                self._loss_angles = loss_free + loss_dist
                
                self._loss_scales = self.basic_loss(pred_scales, self._target_scales)
                
                loss_list = [self._loss_angles,  self._loss_scales]
                
                pred_points = self.osim_model.forward(pred_coords, pred_scales)
                self._loss_points = self.get_MPJPE_L1(pred_points, self._target_points)
                
                # point_loss
                loss_list += [self._loss_points]
                    
                self._loss_angle_range = self.angle_interval_loss(pred_coords, self.angle_range).mean()
                # range_loss
                loss_list += [self._loss_angle_range]
                
                self._loss_all = torch.stack(loss_list).sum()
                
                self.pred_coords = pred_coords[0].cpu().data.numpy()
                self.pred_scales = pred_scales[0].cpu().data.numpy()
                self._reconstruction_error = self.get_MPJPE(pred_points, self._target_points).cpu().data.numpy()
                
        return

    def optimize_parameters(self):
        if self._is_train:
            self._optimizer_img_encoder.zero_grad()
            loss_G = self._forward_G()
            loss_G.backward()
            self._optimizer_img_encoder.step()

    def _forward_G(self):
        self._img_encoder(self._input_image)
        _B = self._input_image.shape[0]
        
        
        pred = self._img_encoder.query_frame(self._input_points, self._input_points_full)
        
        pred_angles = pred[:, :36]
        pred_pelvis_pos = torch.zeros((_B, 3, 1)).float().to(self.device)
        pred_coords = torch.concatenate((pred_angles[:, :3], pred_pelvis_pos, pred_angles[:, 3:]), dim=1)
        pred_scales = pred[:, 36:].reshape(_B, -1, 3) / 8 + 1.0
        
        loss_free = self.get_rot_loss(pred_coords[:, FREE_ROT], self._target_coords[:, FREE_ROT]).mean()
        loss_dist = self.basic_loss(pred_coords[:, NORM_ANGLES], self._target_coords[:, NORM_ANGLES])
        
        self._loss_angles = loss_free + loss_dist
        
        self._loss_scales = self.basic_loss(pred_scales, self._target_scales)
        
        loss_list = [self._loss_angles,  self._loss_scales]
        
        # point_loss
        pred_points = self.osim_model.forward(pred_coords, pred_scales)
        self._loss_points = self.get_MPJPE_L1(pred_points, self._target_points)
        loss_list += [self._loss_points]
            
        # range_loss
        self._loss_angle_range = self.angle_interval_loss(pred_coords, self.angle_range).mean()
        loss_list += [self._loss_angle_range]
        
        self._loss_all = torch.stack(loss_list).sum()
            
        return self._loss_all

    def get_current_errors(self):
        loss_dict = OrderedDict([
                                 ('Angle_loss', self._loss_angles.cpu().data.numpy()),
                                 ('PelvisPos_loss', self._loss_pelvis_pos.cpu().data.numpy()),
                                 ('AngleRange_loss', self._loss_angle_range.cpu().data.numpy()),
                                 ('Distance_loss', self._loss_points.cpu().data.numpy()),
                                 ('Distance_loss_proj', self._loss_points_proj.cpu().data.numpy()),
                                 ('Scale_loss', self._loss_scales.cpu().data.numpy()),
                                 ('Total_loss', self._loss_all.cpu().data.numpy()),
                                 ('Val. OpenSim reconstruction error', self._reconstruction_error),
                                ])
        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        if type(self.pred_coords) == type(None):
            return visuals

        visuals['input_image'] = []
        seg_size = self._input_image.shape[1] // self._opt.num_view
        for i in range(self._opt.num_view):
            view = (self._input_image[0, i * seg_size: (i+1) * seg_size].cpu().data.numpy().transpose(1,2,0)* [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            visuals['input_image'].append(view)
   
        visuals['osim_prediction'] = self.pred_coords
        visuals['osim_groundtruth'] = self._target_coords[0].cpu().data.numpy()

        self.pred_coords = None
        self.pred_scales = None
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
            self._current_lr_G = param_group['lr']
            break