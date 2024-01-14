import torch.nn as nn
import torch.nn.functional as F


class Point2Osim3D(nn.Module):
    def __init__(self, num_view = 1, input_point_dimensions=3, num_input_points=500, reduced_dims=1, frame_pred_dims = 39+22*3, point_feat_dim = 1024):
        super(Point2Osim3D, self).__init__()
        self._num_view = num_view
        
        self.feat_l1 = 512 #point_feat_dim // 2
        self.feat_l2 = 128 #point_feat_dim // 8
        
        if self._num_view == 2:
            self.fc1 = nn.utils.weight_norm(nn.Conv1d(point_feat_dim * 2 + input_point_dimensions, self.feat_l1, kernel_size=1, bias=True))
            self.fc2 = nn.utils.weight_norm(nn.Conv1d(self.feat_l1, self.feat_l1, kernel_size=1, bias=True))
            self.fc3 = nn.utils.weight_norm(nn.Conv1d(self.feat_l1, reduced_dims, kernel_size=1, bias=True))
            
            self.fc4 = nn.utils.weight_norm(nn.Conv1d(num_input_points*reduced_dims, self.feat_l2, kernel_size=1, bias=True))   
            self.fc5 = nn.utils.weight_norm(nn.Conv1d(self.feat_l2, self.feat_l2, kernel_size=1, bias=True))
            self.fc6 = nn.utils.weight_norm(nn.Conv1d(self.feat_l2, frame_pred_dims, kernel_size=1, bias=True))
        else:
            self.fc1 = nn.utils.weight_norm(nn.Conv1d(point_feat_dim + input_point_dimensions, 512, kernel_size=1, bias=True))
            self.fc2 = nn.utils.weight_norm(nn.Conv1d(512, 512, kernel_size=1, bias=True))
            self.fc3 = nn.utils.weight_norm(nn.Conv1d(512, reduced_dims, kernel_size=1, bias=True))
            
            self.fc4 = nn.utils.weight_norm(nn.Conv1d(num_input_points*reduced_dims, 128, kernel_size=1, bias=True))
            self.fc5 = nn.utils.weight_norm(nn.Conv1d(128, 128, kernel_size=1, bias=True))
            self.fc6 = nn.utils.weight_norm(nn.Conv1d(128, frame_pred_dims, kernel_size=1, bias=True))
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
  
        x = x.reshape(x.shape[0], -1, 1)

        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        
        return x
