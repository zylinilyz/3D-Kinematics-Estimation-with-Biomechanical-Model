import torch.nn as nn
from .networks import NetworkBase
import torch
import torch.nn.functional as F
from .HGFilters import HGFilter
from . import geometry
import copy

class Network(NetworkBase):
    def __init__(self, num_view = 1, input_point_dimensions=3, input_channels=3, pred_dimensions=6890):
        super(Network, self).__init__()
        self._name = 'DeformationNet'
        self._num_view = num_view
        self._input_channels = input_channels
        self.image_filter = HGFilter(4, 2, input_channels, 256, 'group', 'no_down', False)

        if self._num_view == 2:
            self.fc1 = nn.utils.weight_norm(nn.Conv1d(1024*2 + input_point_dimensions, 512, kernel_size=1, bias=True))
            self.fc2 = nn.utils.weight_norm(nn.Conv1d(512, 512, kernel_size=1, bias=True))
            self.fc3 = nn.utils.weight_norm(nn.Conv1d(512, pred_dimensions, kernel_size=1, bias=True))
        else:
            self.fc1 = nn.utils.weight_norm(nn.Conv1d(1024 + input_point_dimensions, 512, kernel_size=1, bias=True))
            self.fc2 = nn.utils.weight_norm(nn.Conv1d(512, 512, kernel_size=1, bias=True))
            self.fc3 = nn.utils.weight_norm(nn.Conv1d(512, pred_dimensions, kernel_size=1, bias=True))

        self.frequencies = [  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.]

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
            
            #image_ = torch.cat((image[:, :ch, :, :], image[:, ch:, :, :]),  dim=0)

            assert image_.shape[0] == image.shape[0] * 2, "batch does not match"
            assert image_.shape[1] == image.shape[1] / 2, "channel does not match"

            im_feat_list, normx = self.image_filter(image_)

            self.im_feat_list = []
            self.im_feat_list_view2 = []
            for j, im_feat in enumerate(im_feat_list):
                self.im_feat_list.append(im_feat[range(_B)])
                self.im_feat_list_view2.append(im_feat[range(_B, 2*_B)])

            self.normx = normx[range(_B)]
            self.normx_view2 = normx[range(_B, 2*_B)]

        else:
            self.im_feat_list, self.normx = self.image_filter(image)
            
        return

    def query(self, points):
        # Orthogonal renders are done with scale of 0.2: TODO UPDATE WITH ON-THE-WILD IMAGES WITH WEAK PROJECTION VALS.
        # points: (B, N, 3)
        xy = (points/0.2 + 512)/1024
        xy[:,:,1] = 1-xy[:,:,1]
        xy = xy*2 - 1

        if self._num_view == 2:
            # get xy from view 2
            points_view2 = copy.deepcopy(points)
            # x y z == view0's -z y x
            points_view2[:,:,[0, 1, 2]] = points_view2[:,:,[2, 1, 0]]
            points_view2[:,:,0] = -points_view2[:,:,0]

            xy_view2 = (points_view2/0.2 + 512)/1024
            xy_view2[:,:,1] = 1-xy_view2[:,:,1]
            xy_view2 = xy_view2*2 - 1
        
        points = points/50
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

        x = F.relu(self.fc1(intermediate_preds_list))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def query_test(self, points):
        # More efficient function for testing:
        xy = (points/0.2 + 512)/1024
        xy[:,:,1] = 1-xy[:,:,1]
        xy = xy*2 - 1

        if self._num_view == 2:
            # get xy from view 2
            points_view2 = copy.deepcopy(points)
            # x y z == view0's -z y x
            points_view2[:,:,[0, 1, 2]] = points_view2[:,:,[2, 1, 0]]
            points_view2[:,:,0] = -points_view2[:,:,0]

            xy_view2 = (points_view2/0.2 + 512)/1024
            xy_view2[:,:,1] = 1-xy_view2[:,:,1]
            xy_view2 = xy_view2*2 - 1

        points = points/50
        intermediate_preds_list = points.transpose(2, 1)
        for j, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat, xy)]
            intermediate_preds_list = torch.cat(point_local_feat_list, 1)

        if self._num_view == 2:
            for j, im_feat_view2 in enumerate(self.im_feat_list_view2):
                point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat_view2, xy_view2)]
                intermediate_preds_list = torch.cat(point_local_feat_list, 1)

        x = F.relu(self.fc1(intermediate_preds_list))
        x = F.relu(self.fc2(x))
        if not hasattr(self, 'inference_weights'):
            self.fc3(x) # Run to setup fc3
            self.inference_weights = self.fc3.weight[:,:,0].permute(1,0).reshape(-1,6890,3)
            self.inference_bias = self.fc3.bias.reshape(6890,3)
        x = torch.einsum('fs,fsd->sd', x[0], self.inference_weights) + self.inference_bias

        return x
