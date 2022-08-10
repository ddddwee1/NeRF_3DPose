import torch
import numpy as np
import TorchSUL.Model as M
from lib.utils import kinematics
import pytorch3d.transforms.rotation_conversions as p3dr
 
def axisang_to_rot6d(x):
    print('POSEOPT: Transforming to rotation 6D representation...')
    x = x.reshape(-1, 24, 3)
    rot = p3dr.axis_angle_to_matrix(x)
    return rot[...,:3,:2].flatten(start_dim=-2)

class PoseOptim(M.Model):
    def initialize(self, cfg, pose_metadata):
        self.cfg = cfg
        self.translation = torch.nn.Parameter(torch.tensor(pose_metadata['translation']))
        # self.register_buffer('rest_joints', torch.tensor(pose_metadata['rest_joints']))
        self.rest_joints = torch.nn.Parameter(torch.tensor(pose_metadata['rest_joints']))
        pose = torch.tensor(pose_metadata['poses'])
        if cfg.POSEOPT.use_rot6d:
            pose = axisang_to_rot6d(pose)
        self.poses = torch.nn.Parameter(pose)

        self.kinematics = kinematics.Kinematics(cfg.DATA.smpl_path)

    def forward(self, idx):
        poses = self.poses[idx]
        translation = self.translation[idx]
        rest_joints = self.rest_joints[idx]

        joints, R = self.kinematics(rest_joints, poses, translation, use_rot6d=self.cfg.POSEOPT.use_rot6d)			# [N, J, 3],  [N, J, 3, 3]
        with torch.no_grad():
            dep = joints[:,:,2]					# [N,J]
            near = torch.amin(dep, dim=1)		# [N]
            far = torch.amax(dep, dim=1)		# [N]
            center = (near + far) / 2
            width = (far - near)
            near = center - width * 1.2
            far = center + width * 1.2

        return joints, R, near, far
