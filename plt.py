import pickle 
import torch
import plotly.graph_objects as go
import numpy as np 
from lib.utils import kinematics
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

pairs = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15], [13,16],[14,17],[16,18],[17,19], [18,20],[19,21],\
            [20,22],[21,23],]

# fig = go.Figure()

# pts = pickle.load(open('jts.pkl', 'rb'))
# for p in pairs:
#     x = [pts[p[0],0], pts[p[1],0]]
#     y = [-pts[p[0],1], -pts[p[1],1]]
#     z = [pts[p[0],2], pts[p[1],2]]
#     fig.add_trace(go.Scatter3d(x=x, y=z,z=y, line=dict(color='red', width=15)))

# smps = pickle.load(open('pts.pkl', 'rb'))
# x = smps[:,:,0].reshape(-1)
# y = smps[:,:,1].reshape(-1)
# z = smps[:,:,2].reshape(-1)
# print(x.shape)

# fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='markers'))

# fig.show()

meta = pickle.load(open('dataset/meta.pkl', 'rb'))
idx_sel = np.load('dataset/selected_idx.npy')
K = kinematics.Kinematics('data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
plt.figure(figsize=(10,10))

for i in tqdm(range(len(idx_sel))):
    idx = idx_sel[i]
    mt = meta[idx]
    poses = mt['pose']
    rest_joints = mt['rest_joints']
    translation = mt['translation']
    poses = poses[None, ...]
    rest_joints = rest_joints[None,...]
    translation = translation[None,...]
    poses = torch.from_numpy(poses)
    rest_joints = torch.from_numpy(rest_joints)
    translation = torch.from_numpy(translation)

    joints, R = K(rest_joints, poses, translation)

    joints = joints[0,:,:2] / joints[0,:,2:3]
    joints = joints.numpy() * 10000
    # print(joints)

    img = cv2.imread('./dataset/images/%06d.png'%idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    plt.clf()
    plt.imshow(img)
    for p in pairs:
        x = [joints[p[0],0], joints[p[1],0]]
        y = [joints[p[0],1], joints[p[1],1]]
        plt.plot(x, y)
    plt.savefig('visualized/%06d.jpg'%idx)
