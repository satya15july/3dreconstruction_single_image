import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.path as mplPath
import visualization as viz

floor_3d_pts = np.load("./data/floor_3d_pts.npy")
floor_2d_pts = np.load("./data/floor_2d_pts.npy")

left_facade_3d_pts = np.load("./data/left_facade_3d_pts.npy")
left_facade_2d_pts = np.load("./data/left_facade_2d_pts.npy")

right_facade_3d_pts = np.load("./data/right_facade_3d_pts.npy")
right_facade_2d_pts = np.load("./data/right_facade_2d_pts.npy")

left_facade_roof_3d_pts = np.load("./data/left_facade_roof_3d_pts.npy")
left_facade_roof_2d_pts = np.load("./data/left_facade_roof_2d_pts.npy")

right_facade_roof_3d_pts = np.load("./data/right_facade_roof_3d_pts.npy")
right_facade_roof_2d_pts = np.load("./data/right_facade_roof_2d_pts.npy")

color_pts = np.load("./data/color_pts.npy")

all_3d_pts = np.vstack((floor_3d_pts, left_facade_3d_pts, left_facade_roof_3d_pts, right_facade_3d_pts, right_facade_roof_3d_pts))
all_2d_pts = np.vstack((floor_2d_pts, left_facade_2d_pts, left_facade_roof_2d_pts, right_facade_2d_pts, right_facade_roof_2d_pts))
all_color_pts = np.array(color_pts)

print("all_3d_pts.shape", all_3d_pts.shape)
print("all_2d_pts.shape", all_2d_pts.shape)
print("all_color_pts.shape", all_color_pts.shape)
# Plot actual 3d points
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect('equal')
ax.set_box_aspect([1,1,1])
ax.plot(all_3d_pts[:, 0], all_3d_pts[:, 1], all_3d_pts[:, 2], 'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=140, azim=0)
plt.show()

#utils.visualize_3d(all_3d_pts)
#utils.visualize_3d(pts_3d_2)
with open('3d_reconstruct.ply', 'w') as f:
    # Write the PLY header
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(len(all_3d_pts)))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('end_header\n')

    # Write the vertex data
    for i in range(len(all_3d_pts)):
        f.write('{:.6f} {:.6f} {:.6f} {} {} {}\n'.format(
            all_3d_pts[i, 0], all_3d_pts[i, 1], all_3d_pts[i, 2],
            all_color_pts[i, 0], all_color_pts[i, 1], all_color_pts[i, 2]))
