import numpy as np
import torch
import torch.nn as nn
import copy


class SIMMSpline:
    def __init__(self, x, y, b, c, d):
        self.x = x
        self.y = y
        self.b = b
        self.c = c
        self.d = d
        self.maxIdx = self.b.size(-2) - 1

    def evaluate(self, xs):
        index = torch.bucketize(xs.detach(), self.x) - 1
        index = index.clamp(0, self.maxIdx)

        dx = (xs - self.x[index]).unsqueeze(-1)
        return self.y[index, :] + (self.b[index, :] + (self.c[index, :] + self.d[index, :] * dx) * dx) * dx

class CoordTransformFunction:
    def __init__(self, function_type, scale, spline = None, lin_coeff = None, linear_intercept = None, const = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.type = function_type
        self.spline = spline
        self.lin_coeff = lin_coeff
        self.linear_intercept = linear_intercept
        self.const = const
        self.scale = scale

    def evaluate(self, x):
        if self.type == 'spline':
            out = (self.spline.evaluate(x) * self.scale).squeeze(-1)
        elif self.type == 'linear':
            out = (x * self.lin_coeff + self.linear_intercept) * self.scale
        else:
            out = torch.ones((x.shape)).to(self.device) * self.const * self.scale
        return out

    def to_device(self, device):
        self.device = device
        if self.spline is not None:
            self.spline.x = self.spline.x.to(device)
            self.spline.y = self.spline.y.to(device)
            self.spline.b = self.spline.b.to(device)
            self.spline.c = self.spline.c.to(device)
            self.spline.d = self.spline.d.to(device)

        if self.lin_coeff is not None:
            self.lin_coeff = self.lin_coeff.to(device)
        
        if self.scale is not None:
            self.scale = self.scale.to(device)

        if self.linear_intercept is not None:
            self.linear_intercept = self.linear_intercept.to(device)
        
        if self.const is not None:
            self.const = self.const.to(device)
        

class OSIM(nn.Module):
    def __init__(self, joints_info, body_info, ground_info, coords, coords_range, coord2idx, body2idx):
        super(OSIM, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.joints_info = joints_info
        self.body_info = body_info
        self.ground_info = ground_info
        
        for joint, info in self.joints_info.items():
            for attr, value in info.items():
                if str.split(attr, '_')[-1] == 't':
                    self.joints_info[joint][attr] = self.joints_info[joint][attr].unsqueeze(0).unsqueeze(-1).to(self.device)
    
                if str.split(attr, '_')[-1] == 'r':
                    self.joints_info[joint][attr] = self.joints_info[joint][attr].unsqueeze(0).to(self.device)

                if attr == 'sp_X_func':
                    for k, v in value.items():
                        if k != 'order':
                            _, func = v
                            func.to_device(self.device)

        for body, info in self.body_info.items():
            for attr in info.keys():
                if str.split(attr, '_')[-1] == 't':
                    self.body_info[body][attr] = self.body_info[body][attr].unsqueeze(0).unsqueeze(-1).to(self.device)
                    
        for key in self.ground_info.keys():
            if str.split(key, '_')[-1] == 't':
                self.ground_info[key] = self.ground_info[key].unsqueeze(0).unsqueeze(-1).to(self.device) 
            if str.split(key, '_')[-1] == 'r':
                self.ground_info[key] = self.ground_info[key].unsqueeze(0).to(self.device) 
        
        self.coords = coords.to(self.device) 
        self.coords_range = coords_range.to(self.device) 

        self.coords_range[10] = copy.deepcopy(self.coords_range[9])
        self.coords_range[18] = copy.deepcopy(self.coords_range[17])

        self.coord2idx = coord2idx
        self.body2idx = body2idx
        self.joint2idx = {k: int(i) for i, k in enumerate(self.joints_info.keys())}

        self.joint_chains = [
            ['ground_pelvis']
          , ['hip_r', 'hip_l', 'back']
          , ['patellofemoral_r', 'walker_knee_r', 'acromial_r', 'patellofemoral_l', 'walker_knee_l', 'acromial_l']
          , ['ankle_r', 'elbow_r', 'ankle_l', 'elbow_l']
          , ['subtalar_r', 'radioulnar_r', 'subtalar_l', 'radioulnar_l']
          , ['mtp_r', 'radius_hand_r', 'mtp_l', 'radius_hand_l']
        ]
        
    def forward(self, coords_new, scales_new, isTrain = True):
        # coords_new (B, 39)
        scales_new = torch.abs(scales_new)
        output_pos, output_rot = self.getNewPose_fast(coords_new, scales_new, isTrain=isTrain)
        if isTrain is True:
            return output_pos
        else:
            return output_pos, output_rot

    def euler_angles_to_matrix(self, euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
        """
        Convert rotations given as Euler angles in radians to rotation matrices.
    
        Args:
            euler_angles: Euler angles in radians as tensor of shape (..., 3).
            convention: Convention string of three uppercase letters from
                {"X", "Y", and "Z"}.
    
        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        matrices = [
            self._axis_angle_rotation(c, e)
            for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]
        # return functools.reduce(torch.matmul, matrices)
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

    def _axis_angle_rotation(self, axis: str, angle: torch.Tensor) -> torch.Tensor:
        """
        Return the rotation matrices for one of the rotations about an axis
        of which Euler angles describe, for each value of the angle given.
    
        Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians
    
        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
    
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
    
        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")
    
        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def getNewPose(self, coords_new, scales_new):

        body_info = {}
        body_pos = []
        joint_pos = []
        B = coords_new.shape[0]
        #BpJp_time = 0
        #GJp_time = 0
        #JpJc_time = 0
        #GJc_time = 0
        #BcJc_time = 0
        #GBc_time = 0
        #BBcen_time = 0
        #GBcen_time = 0

        for joint in self.joints_info.keys():
            #sp_X_func = self.joints_info[joint]['sp_X_func']
            
            p_body_name = self.joints_info[joint]['p_body_name']
            scale_p = scales_new[:, self.body2idx[p_body_name]] if p_body_name != 'ground' else torch.ones((B, 3)).to(self.device)
            scale_p = scale_p.unsqueeze(-1)#[..., None]
            
            c_body_name = self.joints_info[joint]['c_body_name']
            scale_c = scales_new[:, self.body2idx[c_body_name]]
            scale_c = scale_c.unsqueeze(-1)#[..., None]

            # get Bp info
            if p_body_name == 'ground':
                X_GBp_r = self.ground_info['X_G_r']
                X_GBp_t = self.ground_info['X_G_t']
            else:
                X_GBp_r = body_info[p_body_name]['X_GB_r']
                X_GBp_t = body_info[p_body_name]['X_GB_t']

            # get Jp info
            # - get BpJp
            #s = time.time()
            X_BpJp_r = self.joints_info[joint]['X_BpJp_r']
            X_BpJp_t = self.joints_info[joint]['X_BpJp_t']
            #e = time.time()
            #BpJp_time += (e - s)
            
            # - get Jp
            #s = time.time()
            X_GJp_r_new = torch.matmul(X_GBp_r, X_BpJp_r)
            X_GJp_t_new = X_GBp_t + torch.matmul(X_GBp_r, torch.mul(X_BpJp_t, scale_p))

            if p_body_name != 'ground':
                joint_pos.append(X_GJp_t_new)

            #e = time.time()
            #GJp_time += (e - s)

            # get JpJc
            #s = time.time()
            X_JpJc_t_new = torch.zeros(X_GJp_t_new.shape).to(self.device)
            X_JpJc_r_new = torch.zeros(X_GJp_t_new.shape).to(self.device)

            sp_X_rot_func = self.joints_info[joint]['sp_X_rot_func']
            sp_X_rot_coord = self.joints_info[joint]['sp_X_rot_coord']
            sp_X_rot_order = self.joints_info[joint]['sp_X_rot_order']
            sp_X_trans_func = self.joints_info[joint]['sp_X_trans_func']
            sp_X_trans_coord = self.joints_info[joint]['sp_X_trans_coord']

            for iter in range(3):
                xs = coords_new[:, self.coord2idx[sp_X_trans_coord[iter]]] 
                X_JpJc_t_new[:, iter, :] = sp_X_trans_func[iter].evaluate(xs)#[..., None]
                xs = coords_new[:, self.coord2idx[sp_X_rot_coord[iter]]] 
                X_JpJc_r_new[:, iter, :] = sp_X_rot_func[iter].evaluate(xs)#[..., None]


            X_JpJc_r_new = self.euler_angles_to_matrix(X_JpJc_r_new.squeeze(-1), convention=sp_X_rot_order)

            #e = time.time()
            #JpJc_time += (e - s)
            #print('JpJc', e - s)

            # get Jc
            #s = time.time()
            X_GJc_r_new = torch.matmul(X_GJp_r_new, X_JpJc_r_new)
            X_GJc_t_new = X_GJp_t_new + torch.matmul(X_GJp_r_new,torch.mul(X_JpJc_t_new, scale_p))
            
            joint_pos.append(X_GJc_t_new)

            #e = time.time()
            #GJc_time += (e - s)
            
            # get Bc
            #s = time.time()
            X_BcJc_t = self.joints_info[joint]['X_BcJc_t']
            X_JcBc_r = self.joints_info[joint]['X_JcBc_r']
            #e = time.time()
            #BcJc_time += (e - s)
            
            #s = time.time()
            X_GBc_r_new = torch.matmul(X_GJc_r_new, X_JcBc_r)
            X_GBc_t_new = X_GJc_t_new - torch.matmul(X_GBc_r_new, torch.mul(X_BcJc_t, scale_c))

            body_info[c_body_name] = {'X_GB_r': X_GBc_r_new, 'X_GB_t': X_GBc_t_new}
            
            #e = time.time()
            #GBc_time += (e - s)

            # get Bcenter
            #s = time.time()
            X_BBcen_t = self.body_info[c_body_name]['X_BBcen_t']
            #e = time.time()
            #BBcen_time += (e - s)

            #s = time.time()
            X_GBcen_t_new = X_GBc_t_new + torch.matmul(X_GBc_r_new, torch.mul(X_BBcen_t, scale_c))

            body_pos.append(X_GBcen_t_new)
            #e = time.time()
            #GBcen_time += (e - s)

        body_pos = torch.stack(body_pos, 0).squeeze(-1).permute(1, 0, 2)
        joint_pos = torch.stack(joint_pos, 0).squeeze(-1).permute(1, 0, 2)
        output_pos = torch.cat((body_pos, joint_pos), dim=1)

        #print('BpJp_time' , BpJp_time)
        #print('GJp_time', GJp_time)
        #print('JpJc_time', JpJc_time)
        #print('GJc_time', GJc_time)
        #print('BcJc_time', BcJc_time)
        #print('GBc_time', GBc_time)
        #print('BBcen_time', BBcen_time)
        #print('GBcen_time', GBcen_time)

        return output_pos

    def getNewPose_fast(self, coords_new, scales_new, isTrain = True):

        B = coords_new.shape[0]
        body_info = {}

        body_pos_list = []
        body_rot_list = []
        body_pos_n = []
        joint_pos_list = []
        joint_rot_list = []
        joint_pos_n = []

        #BpJp_time = 0
        #GJp_time = 0
        #JpJc_time = 0
        #GJc_time = 0
        #BcJc_time = 0
        #GBc_time = 0
        #BBcen_time = 0
        #GBcen_time = 0

        for hier in self.joint_chains:

            X_GBp_t_hier = []
            X_GBp_r_hier = []
            X_BpJp_t_hier = []
            X_BpJp_r_hier = []
            scale_p_hier = []
            X_JpJc_t_hier = []
            X_JpJc_r_hier = []
            X_BcJc_t_hier = []
            X_JcBc_r_hier = []
            scale_c_hier = []
            X_BBcen_t_hier = []

            for joint in hier:
                p_body_name = self.joints_info[joint]['p_body_name']
                scale_p = scales_new[:, self.body2idx[p_body_name]] if p_body_name != 'ground' else torch.ones((B, 3)).to(self.device)
                scale_p = scale_p.unsqueeze(-1)#[..., None]
                scale_p_hier.append(scale_p)
                
                c_body_name = self.joints_info[joint]['c_body_name']
                scale_c = scales_new[:, self.body2idx[c_body_name]]
                scale_c = scale_c.unsqueeze(-1) #[..., None]
                scale_c_hier.append(scale_c)
    
                #### get Bp ####
                if p_body_name == 'ground':
                    X_GBp_r = self.ground_info['X_G_r']
                    X_GBp_t = self.ground_info['X_G_t']

                    X_GBp_r_hier.append(X_GBp_r.repeat(B, 1, 1))
                    X_GBp_t_hier.append(X_GBp_t.repeat(B, 1, 1))
                else:
                    X_GBp_r = body_info[p_body_name]['X_GB_r']
                    X_GBp_t = body_info[p_body_name]['X_GB_t']
    
                    X_GBp_r_hier.append(X_GBp_r)
                    X_GBp_t_hier.append(X_GBp_t)

                #### get BpJp ####
                #s = time.time()
                X_BpJp_r = self.joints_info[joint]['X_BpJp_r']
                X_BpJp_t = self.joints_info[joint]['X_BpJp_t']

                X_BpJp_r_hier.append(X_BpJp_r.repeat(B, 1, 1))
                X_BpJp_t_hier.append(X_BpJp_t.repeat(B, 1, 1))
                
                #e = time.time()
                #BpJp_time += (e - s)
    
                #### get JpJc ####
                #s = time.time()
                X_JpJc_t_new = torch.zeros((B, 3, 1)).to(self.device)
                X_JpJc_r_new = torch.zeros((B, 3, 1)).to(self.device)
    
                sp_X_rot_func = self.joints_info[joint]['sp_X_rot_func']
                sp_X_rot_coord = self.joints_info[joint]['sp_X_rot_coord']
                sp_X_rot_order = self.joints_info[joint]['sp_X_rot_order']
                sp_X_trans_func = self.joints_info[joint]['sp_X_trans_func']
                sp_X_trans_coord = self.joints_info[joint]['sp_X_trans_coord']
    
                for iter in range(3):
                    xs = coords_new[:, self.coord2idx[sp_X_trans_coord[iter]]] 
                    X_JpJc_t_new[:, iter, :] = sp_X_trans_func[iter].evaluate(xs)#[..., None]
                    xs = coords_new[:, self.coord2idx[sp_X_rot_coord[iter]]] 
                    X_JpJc_r_new[:, iter, :] = sp_X_rot_func[iter].evaluate(xs)#[..., None]

                X_JpJc_r_new = self.euler_angles_to_matrix(X_JpJc_r_new.squeeze(-1), convention=sp_X_rot_order)
    
                X_JpJc_r_hier.append(X_JpJc_r_new)
                X_JpJc_t_hier.append(X_JpJc_t_new)

                #e = time.time()
                #JpJc_time += (e - s)
    
                #### get Bc ####
                #s = time.time()

                X_BcJc_t = self.joints_info[joint]['X_BcJc_t']
                X_JcBc_r = self.joints_info[joint]['X_JcBc_r']

                X_JcBc_r_hier.append(X_JcBc_r.repeat(B, 1, 1))
                X_BcJc_t_hier.append(X_BcJc_t.repeat(B, 1, 1))
                
                #e = time.time()
                #BcJc_time += (e - s)
    
                #### get Bcenter ####
                #s = time.time()
                X_BBcen_t = self.body_info[c_body_name]['X_BBcen_t']
                X_BBcen_t_hier.append(X_BBcen_t.repeat(B, 1, 1))
                #e = time.time()
                #BBcen_time += (e - s)
        
            # stack
            X_GBp_t_hier   = torch.cat(X_GBp_t_hier  , dim=0)
            X_GBp_r_hier   = torch.cat(X_GBp_r_hier  , dim=0)
            X_BpJp_t_hier  = torch.cat(X_BpJp_t_hier , dim=0)
            X_BpJp_r_hier  = torch.cat(X_BpJp_r_hier , dim=0)
            scale_p_hier   = torch.cat(scale_p_hier  , dim=0)
            X_JpJc_t_hier  = torch.cat(X_JpJc_t_hier , dim=0)
            X_JpJc_r_hier  = torch.cat(X_JpJc_r_hier , dim=0)
            X_BcJc_t_hier  = torch.cat(X_BcJc_t_hier , dim=0)
            X_JcBc_r_hier  = torch.cat(X_JcBc_r_hier , dim=0)
            scale_c_hier   = torch.cat(scale_c_hier  , dim=0)
            X_BBcen_t_hier = torch.cat(X_BBcen_t_hier, dim=0)
    
            # calculate
            # Jp
            #s = time.time()
            X_GJp_r_new_hier = torch.matmul(X_GBp_r_hier, X_BpJp_r_hier)
            X_GJp_t_new_hier = X_GBp_t_hier + torch.matmul(X_GBp_r_hier, torch.mul(X_BpJp_t_hier, scale_p_hier))
            
            ###for i, joint in enumerate(hier):
            ###    if self.joints_info[joint]['p_body_name'] != 'ground':
            ###        joint_pos_list.append(X_GJp_t_new_hier[B*i:B*(i+1)])
            ###        joint_pos_n.append(2*(self.joint2idx[joint])-1)
            #e = time.time()
            #GJp_time += (e - s)
    
            # Jc
            #s = time.time()
            X_GJc_r_new_hier  = torch.matmul(X_GJp_r_new_hier, X_JpJc_r_hier)
            X_GJc_t_new_hier  = X_GJp_t_new_hier  + torch.matmul(X_GJp_r_new_hier, torch.mul(X_JpJc_t_hier, scale_p_hier))
            
            for i, joint in enumerate(hier):
                joint_pos_list.append(X_GJc_t_new_hier[B*i:B*(i+1)])
                joint_rot_list.append(X_GJc_r_new_hier[B*i:B*(i+1)])
                joint_pos_n.append(2*(self.joint2idx[joint]))

            #e = time.time()
            #GJc_time += (e - s)
    
            # Bc
            #s = time.time()
            X_GBc_r_new_hier = torch.matmul(X_GJc_r_new_hier, X_JcBc_r_hier)
            X_GBc_t_new_hier = X_GJc_t_new_hier - torch.matmul(X_GBc_r_new_hier, torch.mul(X_BcJc_t_hier, scale_c_hier))
            
            #e = time.time()
            #GBc_time += (e - s)

            # Bcen
            #s = time.time()
            X_GBcen_t_new_hier = X_GBc_t_new_hier + torch.matmul(X_GBc_r_new_hier, torch.mul(X_BBcen_t_hier, scale_c_hier))
            for i, joint in enumerate(hier):
                c_body_name = self.joints_info[joint]['c_body_name']
                if isTrain is True:
                    body_pos_list.append(X_GBcen_t_new_hier[B*i:B*(i+1)])
                else:
                    body_pos_list.append(X_GBc_t_new_hier[B*i:B*(i+1)])
                body_rot_list.append(X_GBc_r_new_hier[B*i:B*(i+1)])
                body_pos_n.append(self.body2idx[c_body_name])
                body_info[c_body_name] = {'X_GB_r': X_GBc_r_new_hier[B*i:B*(i+1)], 'X_GB_t': X_GBc_t_new_hier[B*i:B*(i+1)]}
            
            #e = time.time()
            #GBcen_time += (e - s)
                
        #body_pos = torch.stack(body_pos, 0).squeeze(-1).permute(1, 0, 2)
        #joint_pos = torch.stack(joint_pos, 0).squeeze(-1).permute(1, 0, 2)
        
        body_pos = []
        body_rot = []
        sort_index = np.argsort(body_pos_n)
        for idx in sort_index:
            body_pos.append(body_pos_list[idx])
            body_rot.append(body_rot_list[idx])
            

        body_pos = torch.stack(body_pos, 0).squeeze(-1).permute(1, 0, 2)
        body_rot = torch.stack(body_rot, 0).permute(1, 0, 2, 3)
        #body_pos = torch.cat(body_pos, dim=0)
        #body_pos = body_pos.squeeze(-1)

        joint_pos = []
        joint_rot = []
        sort_index = np.argsort(joint_pos_n)
        for idx in sort_index:
            joint_pos.append(joint_pos_list[idx])
            joint_rot.append(joint_rot_list[idx])
        
        joint_pos = torch.stack(joint_pos, 0).squeeze(-1).permute(1, 0, 2)
        joint_rot = torch.stack(joint_rot, 0).permute(1, 0, 2, 3)
        #joint_pos = torch.cat(joint_pos, dim=0)
        #joint_pos = joint_pos.squeeze(-1)
        
        output_pos = torch.cat((body_pos, joint_pos), dim=1)
        output_rot = torch.cat((body_rot, joint_rot), dim=1)

        #print('BpJp_time' , BpJp_time)
        #print('GJp_time', GJp_time)
        #print('JpJc_time', JpJc_time)
        #print('GJc_time', GJc_time)
        #print('BcJc_time', BcJc_time)
        #print('GBc_time', GBc_time)
        #print('BBcen_time', BBcen_time)
        #print('GBcen_time', GBcen_time)


        return output_pos, output_rot



