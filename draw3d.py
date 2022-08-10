import plotly.graph_objects as go
import numpy as np 
import pickle 
from easydict import EasyDict
import yaml

cfg = EasyDict(yaml.load(open('config.yaml'), Loader=yaml.FullLoader))

index = 0
grid_values, grids = pickle.load(open(f'{cfg.EXPERIMENT.name}/grid_data/{index:06d}.pkl', 'rb'))

grid_x, grid_y, grid_z = grids
grid_y, grid_z = grid_z, grid_y 
grid_z = - grid_z

pairs = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15], [13,16],[14,17],[16,18],[17,19], [18,20],[19,21],\
            [20,22],[21,23],]

fig = go.Figure(data=go.Volume(
    x=grid_x.flatten(),
    y=grid_y.flatten(),
    z=grid_z.flatten(),
    value=grid_values.flatten(),
    isomin=0,
    isomax=2,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=300, # needs to be a large number for good volume rendering
    ))

# pts = pickle.load(open('pts/%06d.pkl'%index, 'rb'))
# for p in pairs:
#     x = [pts[p[0],0], pts[p[1],0]]
#     y = [-pts[p[0],1], -pts[p[1],1]]
#     z = [pts[p[0],2], pts[p[1],2]]
#     fig.add_trace(go.Scatter3d(x=x, y=z,z=y, line=dict(color='red', width=15)))

# pts = pickle.load(open('pts_pred/%06d.pkl'%index, 'rb'))
# for p in pairs:
#     x = [pts[p[0],0], pts[p[1],0]]
#     y = [-pts[p[0],1], -pts[p[1],1]]
#     z = [pts[p[0],2], pts[p[1],2]]
#     fig.add_trace(go.Scatter3d(x=x, y=z,z=y, line=dict(color='green', width=15)))

# pairs = [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8], [8,9], [9,10], [8,11], [11,12], [12,13], [8,14], [14,15], [15,16]]

# pts = pickle.load(open('pts_gt/%06d.pkl'%index, 'rb'))
# for p in pairs:
#     x = [pts[p[0],0], pts[p[1],0]]
#     y = [-pts[p[0],1], -pts[p[1],1]]
#     z = [pts[p[0],2], pts[p[1],2]]
#     fig.add_trace(go.Scatter3d(x=x, y=z,z=y, line=dict(color='yellow', width=15)))

fig.show()

