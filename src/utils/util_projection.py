import torch


def project_by_camera_projection(joints_cv, pro_side):
    """
    Reference: https://github.com/google-research-datasets/Objectron/blob/master/notebooks/objectron-geometry-tutorial.ipynb
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    function project_points
    """
    vertices_3d = joints_cv.reshape(joints_cv.shape[0], -1,3)
    vertices_3d_homg = torch.concatenate((vertices_3d, torch.ones_like(vertices_3d[:, :, :1])), axis=-1).permute(0, 2, 1)
    # vertices_2d_proj (F, 3, 1)
    vertices_2d_proj = torch.matmul(pro_side.unsqueeze(0), vertices_3d_homg)
    # Project the points
    points2d_ndc = vertices_2d_proj[:, :-1, :] / vertices_2d_proj[:, -1:, :]
    points2d_ndc = points2d_ndc.permute(0, 2, 1)
    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    arranged_points = points2d_ndc[:, :,:2]
    return arranged_points

def proj_3d_to_2d(point_3d, pro_front, meta_front):
    # F, 500, 3
    points_2d = project_by_camera_projection(point_3d, pro_front)

    # (F, 1, 1)
    scales = meta_front[:, 2].unsqueeze(-1).unsqueeze(-1)
    # (F, 1)
    offset_x = meta_front[:, 0].unsqueeze(-1)
    offset_y = meta_front[:, 1].unsqueeze(-1)

    points_2d[:, :, 0] -= offset_x
    points_2d[:, :, 1] -= offset_y
    points_2d = points_2d / scales

    points_2d = torch.clip(points_2d, 0, 255)
    
    return points_2d