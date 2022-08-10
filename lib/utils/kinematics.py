import torch
import pickle
import numpy as np
import TorchSUL.Model as M
import torch.nn.functional as F
import pytorch3d.transforms.rotation_conversions as p3dr

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (*,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x_shape = x.shape[:-1]
    x = x.reshape(-1,3,2)

    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1).reshape(*x_shape, 3, 3)


class Kinematics(M.Model):

    def initialize(self, model_dir):
        smpl_model = pickle.load(open(model_dir, 'rb'), encoding='latin1')
        ktree = smpl_model['kintree_table']
        parent_col_dict = {ktree[1, i]: i for i in range(ktree.shape[1])}  # correspodance id -> column
        parent_arr = np.array([parent_col_dict[ktree[0, i]] for i in range(1, ktree.shape[1])])
        self.add_buffer('parent', parent_arr)

    def add_buffer(self, name, array):
        self.register_buffer(name, torch.from_numpy(array))

    def forward(self, J, pose, translation, use_rot6d=False):
        if use_rot6d:
            R = rot6d_to_rotmat(pose)
        else:
            pose = pose.reshape(-1, 24, 3)
            R = p3dr.axis_angle_to_matrix(pose)  # [N, J, 3, 3]

        J_rel = J[:, 1:] - J[:, self.parent]
        J_rel = torch.cat([J[:, 0:1], J_rel], dim=1)  # [N, J, 3]
        J_rel = torch.cat([J_rel, torch.ones(J_rel.shape[0], J_rel.shape[1], 1, device=J_rel.device)], dim=-1)  # [N, J, 4]
        R_homo = torch.cat([R, torch.zeros(R.shape[0], R.shape[1], 1, 3, device=R.device)], dim=-2)  # [N, J, 4, 3]
        rel_homo = torch.cat([R_homo, J_rel.unsqueeze(-1)],
                             dim=-1)  # [N, J, 4, 4]
        homo = [rel_homo[:, 0]]
        for jj in range(1, rel_homo.shape[1]):
            homo.append(homo[self.parent[jj - 1]] @ rel_homo[:, jj])
        homo = torch.stack(homo, dim=1)  # [N, J, 4, 4]
        joints = homo[:, :, :3, 3]  # [N, J, 3]
        joints = joints + translation.reshape(-1, 1, 3)

        R = homo[:, :, :3, :3]  # [N, J, 3, 3]
        return joints, R
