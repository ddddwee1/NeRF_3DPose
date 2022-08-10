import torch 
import TorchSUL.Model as M

class Encoder(M.Model):
	def forward(self, pts, viewdir, joints, R):
		# pts: [N, Ray, sample, 3]
		# viewdir: [N, ray, 3]
		# R: [N, J, 3, 3]
		# joints: [N, J, 3]

		# print(pts.dtype, viewdir.dtype, joints.dtype, R.dtype)

		diff = pts.unsqueeze(-2) - joints.unsqueeze(1).unsqueeze(1)														# [N, ray, sample, J, 3]
		# Original rotation: Local to world 
		# Here: Reverse the rotation matrix from 'ba' to 'ab', means world to local
		diff_local = torch.einsum('nrsja,njab->nrsjb', diff, R)
		pose_norm = torch.linalg.norm(diff_local, dim=-1, keepdim=True)													# [N, ray, sample, J, 1]
		pose_unit = diff_local / pose_norm																				# [N, ray, sample, J, 3]
		view_local = torch.einsum('nra,njab->nrjb', viewdir, R).unsqueeze(2).expand(-1, -1, pts.shape[2], -1, -1)		# [N, ray, sample, J, 3]
		view_local = view_local / torch.linalg.norm(view_local, dim=-1, keepdim=True)

		J = joints.shape[1]
		N, num_ray, num_sample = pts.shape[0], pts.shape[1], pts.shape[2]

		pose_unit = pose_unit.reshape(N, num_ray, num_sample, J, 3)
		pose_norm = pose_norm.reshape(N, num_ray, num_sample, J, 1)
		view_local = view_local.reshape(N, num_ray, num_sample, J, 3)

		return pose_unit, pose_norm, view_local
