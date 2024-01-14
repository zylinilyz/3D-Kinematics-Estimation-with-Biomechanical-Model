from .networks import NetworkBase
import torch
from .HGFilters import HGFilter
from . import geometry
import copy

from .model_restormer import *
from .model_point2osim_3d import *
from einops import rearrange
from .model_HGHeads import *

class Network(NetworkBase):
    def __init__(self, seq_length = 11, seq_pred_dims = 39+22*3, num_view = 1, input_point_dimensions=2, 
                       input_channels=3, num_input_points=500, reduced_dims=1, frame_pred_dims = 39+22*3, hg_heads=0,
                       hg_out = 256):
        
        super(Network, self).__init__()
        self._num_view = num_view
        self._input_channels = input_channels
        self.hg_heads = hg_heads
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.image_filter = HGFilter(4, 2, input_channels, hg_out, 'group', 'no_down', False)
        
        self.frame2seq = Restormer(inp_channels=1, out_channels=1, dim = int(3*16), num_blocks = [4,6,6], num_refinement_blocks = 1,
                                heads = [1,2,4],ffn_expansion_factor = 2.66, bias = True, LayerNorm_type = 'WithBias',
                                in_feat_len = frame_pred_dims, out_feat_len = seq_pred_dims, group = frame_pred_dims)
    
        point_feat_dim = self.hg_heads * 4 if self.hg_heads > 0 else hg_out * 4
        self.point2osim = Point2Osim3D(num_view = num_view, input_point_dimensions=input_point_dimensions,
                                        num_input_points=num_input_points, reduced_dims=reduced_dims,
                                        frame_pred_dims = frame_pred_dims, point_feat_dim = point_feat_dim)
        
        
        self.to_one_scales = torch.nn.Conv1d(in_channels=seq_length, out_channels=1, kernel_size=1)
        
        self.heads = None
        if self.hg_heads > 0:
            self.heads = HGHeads(in_dim = 256, out_dim = self.hg_heads)
        
        

    def forward(self, image):
        # image could have multi-view for one frame
        # image: (B, num_view*C, H, W)

        # deal with two views
        if self._num_view == 2:
            # reform image
            # from (B, num_view*C, H, W) to (num_view*B, C, H, W)
            _B = image.shape[0]

            # first view
            ch = self._input_channels
            image_ = image[:, 0:ch, :, :].clone()
            
            for i in range(1, self._num_view):
                image_ = torch.cat((image_, image[:, 0+ch*i : ch+ch*i, :, :]), dim=0)
            
            assert image_.shape[0] == image.shape[0] * 2, "batch does not match"
            assert image_.shape[1] == image.shape[1] / 2, "channel does not match"
            
            im_feat_list, normx = self.image_filter(image_)
            
            if self.hg_heads > 0:
                im_feat_list = self.heads(im_feat_list)

            self.im_feat_list = []
            self.im_feat_list_view2 = []
            for j, im_feat in enumerate(im_feat_list):
                self.im_feat_list.append(im_feat[range(_B)])
                self.im_feat_list_view2.append(im_feat[range(_B, 2*_B)])

            self.normx = normx[range(_B)]
            self.normx_view2 = normx[range(_B, 2*_B)]

        else:
            self.im_feat_list, self.normx = self.image_filter(image)
            
            if self.hg_heads > 0:
                self.im_feat_list = self.heads(self.im_feat_list)
            
        return

    def clean_feat(self):
        del self.im_feat_list, self.normx
        if self._num_view == 2:
            del self.im_feat_list_view2, self.normx_view2
        torch.cuda.empty_cache()


    def query_frame(self, points_in, points_full):
            
        # Orthogonal renders are done with scale of 0.2: TODO UPDATE WITH ON-THE-WILD IMAGES WITH WEAK PROJECTION VALS.
        # points: (BF, 2, N, 2), xy: (BF, N, 2)
        xy = points_in[:, 0, :, :]/256
        xy = xy*2 - 1

        if self._num_view == 2:
            xy_view2 = points_in[:, 1, :, :]/256
            xy_view2 = xy_view2*2 - 1
        
        # points -> (BF, N, 2)
        #b bf, n, 3
        points = points_full
        
        intermediate_preds_list = torch.cat([points.transpose(2, 1)], dim=1)
        
        
        # self.im_feat_list is a list with four types of features, 
        # each type have (B, 128, 128, 256)
        for j, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat, xy)]
            intermediate_preds_list = torch.cat(point_local_feat_list, 1)
        
        if self._num_view == 2:
            for j, im_feat_view2 in enumerate(self.im_feat_list_view2):
                point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat_view2, xy_view2)]
                intermediate_preds_list = torch.cat(point_local_feat_list, 1)

        x = self.point2osim(intermediate_preds_list)
        
        return x
    
    def query_frame_3d(self, points):
        # Orthogonal renders are done with scale of 0.2: TODO UPDATE WITH ON-THE-WILD IMAGES WITH WEAK PROJECTION VALS.
        # points: (B, N, 3)
        xy = points/256
        xy[:,:,1] = 1-xy[:,:,1]
        xy = xy*2 - 1

        if self._num_view == 2:
            # get xy from view 2
            points_view2 = copy.deepcopy(points)
            # x y z == view0's -z y x
            points_view2[:,:,[0, 1, 2]] = points_view2[:,:,[2, 1, 0]]
            points_view2[:,:,2] = - points_view2[:,:,2]
            points_view2[:,:,0] = 255 + points_view2[:,:,0]

            xy_view2 = points_view2/256
            xy_view2[:,:,1] = 1-xy_view2[:,:,1]
            xy_view2 = xy_view2*2 - 1
        
        points = points/256
        # points -> (B, 3, N)
        intermediate_preds_list = points.transpose(2, 1)
        # self.im_feat_list is a list with four types of features, 
        # each type have (B, 128, 128, 256)
        for j, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat, xy)]
            intermediate_preds_list = torch.cat(point_local_feat_list, 1)
        
        if self._num_view == 2:
            for j, im_feat_view2 in enumerate(self.im_feat_list_view2):
                point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat_view2, xy_view2)]
                intermediate_preds_list = torch.cat(point_local_feat_list, 1)

        x = self.point2osim(intermediate_preds_list)

        return x

    def query_sequence(self, x):
        x = rearrange(x, 'b f p c -> b c p f')
        x = self.frame2seq(x)
            
        # to one set of scales
        f = x.shape[1]
        c = x[:, :, :36, :]
        s = x[:, :, 36:, :].squeeze(-1)
        
    
        s = self.to_one_scales(s)
        ###s = self.scales_head(s).unsqueeze(1)
        s = s.repeat(1, f, 1).unsqueeze(-1)
        x = torch.concatenate((c, s), dim=2)

        return x

