import itertools
import numpy as np
import sys

from numpy.core.fromnumeric import product
np.set_printoptions(threshold=sys.maxsize)
import cv2

import torch
import time
#import plotly.express as px
import pandas as pd
from numba import jit, float64, int64
import os
import math
import matplotlib
# matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba
from numba import jit, cuda
from timeit import default_timer as timer

 
fx = 636.6593017578125
fy = 636.251953125

cubicSz = 400
croppedSz = 64 #88
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

def ReadDepthMap(path):

    #img = cv2.imread(path) 
    #depth_scale = 0.12498664727900177
    #dpt = img[:, :, 2] + img[:, :, 1] * 256
    #dpt = dpt * depth_scale
    dpt = cv2.imread(path,-1).astype(np.float)

    return dpt

def DepthMapDisplay(img):
    plt.imshow( img, 'gray')
    plt.show()
    ch = cv2.waitKey(0)
    if ch == ord('q'):
       exit(0)

def pixel2world(x, y, z, img_width, img_height, fx, fy):
    # From Similar Triangles u / fx = x / z
    w_x = (x - img_width / 2) * z / fx 
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z

def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy) 
    return points

def PointsDisplay(points):
    x = points[:,:,0].flatten()
    y = points[:,:,1].flatten()
    z = points[:,:,2].flatten()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(z, x, y, c=z, edgecolors='b')
    plt.show()

def discretize(coord, croppedSz):
    '''[-1, 1] -> [0, croppedSz]'''
    min_normalized = -1
    max_normalized = 1
    scale = float (max_normalized - min_normalized) / croppedSz
    return (coord - min_normalized) / scale

def projective_dtsdf_Z_optimized(coord, cropped_size, device, dtype):


    # Removing all points that are far from the center by cropping
    coord = coord.type(torch.int64)

    # print(coord[1])
    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)
    
    coord = coord[mask, :]

    # a tensor of size 88x88 with values 250
    img_z = torch.full((cropped_size, cropped_size), 250.0, dtype=dtype)
    
    # The z value for each point in the coord [x, y] = z
    img_z[coord[:, 0], coord[:, 1]] = coord[:, 2].type(dtype)
    # print(img_z.unsqueeze(2).shape)
    
    Pc = img_z.unsqueeze(2).repeat(1, 1, cropped_size) #(1,1,88)

    # print(Pc.shape)
    _, _, Vc_z = torch.meshgrid(
                torch.arange(start = 0.5, end = cropped_size, step = 1), 
                torch.arange(start = 0.5 ,end = cropped_size, step = 1), 
                torch.arange(start = 0.5, end = cropped_size, step = 1) ,
                )
    Vc_z = Vc_z.to(dtype=dtype)
    

    u = torch.tensor(3.0, dtype = dtype)
    sdf_z = torch.abs(Vc_z-Pc)

    mask_tsdf = torch.ones_like(sdf_z)

    # Multiply points where Vc_z > Pc by -1
    mask_tsdf[torch.gt(Vc_z,Pc)] = -1
    sdf = sdf_z.mul(mask_tsdf)

    # all-ones tensor
    onez_tensor = torch.abs(mask_tsdf)
    
    # Truncate all values < -1 and > 1 to [-1, 1] after dividing by 3
    tsdf = torch.min(torch.max(torch.div(sdf, u), -1 * onez_tensor), onez_tensor)
    # print(tsdf)    
    return tsdf


def TsdfDisplay_modified(cubic, croppedSz):
    x,y,z,colors = [[]for l in range (4)]
    c=1
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=90, azim=-90)

    import itertools
    for i,j,k in itertools.product(range(croppedSz),range(croppedSz),range(croppedSz)):

        if cubic[i,j,k] < 1 and cubic[i,j,k] > -1: # visualize INDICES of voxels within range (-1 -- 1)
            #print(cubic[i,j,k] < 1, 'less than 1' ,  cubic[i,j,k] > -1, 'greater than 1' )
            x.append(i)
            y.append(j)
            z.append(k)
            colors = cubic[i,j,k]

    ax.scatter(x,y,z, c=z, cmap='jet') #NOTE: To visualize actual voxel values use ax.voxels(..) with python v3+
    plt.show()


def showVoxelizedDepth(verts, center = None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]

    ax.scatter(x, y, z)

    if center is not None:
        ax.scatter(center[0], center[1], center[2], c='r', s=500)

    ax.view_init(elev=90, azim=-90)
    plt.show()

def colorVoxelGrid(coord, colors, cropped_size=88):
    
    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)
    
    coord = coord[mask, :]
    colors = colors[mask, :]
    
    color_grid = np.zeros((cropped_size, cropped_size, 3))
    # print(color_grid)
    discrete_points = coord.astype(int)
    for i, p in enumerate(discrete_points):
        color_grid[p[0], p[1]] = colors[i]

    # color_grid.unsqueeze(3)
    color_grid = np.repeat(color_grid[:, :, np.newaxis, :], cropped_size, axis=2)
    return color_grid

def fuseTSDFwithColor(tsdf, color_grid):
    color_grid = torch.tensor(color_grid).permute(3, 0, 1, 2)
    return torch.cat((tsdf, color_grid))

def run_tsdf(path, ref_point, tsdf_channels=1):
    # print(path)
    depthmap = ReadDepthMap(path)
    
    # print(depthmap.shape)
    points = depthmap2points(depthmap, fx, fy)
    points[:,:,1] *= -1

    points = points.reshape((-1, 3)) # flatten 480 x 640 to 307200

    points = (points - ref_point) / (cubicSz / 2)
    points = discretize(points, croppedSz)
    tsdf = projective_dtsdf_Z_optimized(torch.as_tensor(points), croppedSz, device, dtype)
    tsdf = torch.unsqueeze(tsdf, 0)
    
    if tsdf_channels == 4:
        rgb_path = path.replace('depth', 'rgb')#.replace('png', 'jpg')    
        img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        colors = img.reshape((-1, 3)) / 255
        colorGrid = colorVoxelGrid(points, colors, croppedSz)
        tsdf = fuseTSDFwithColor(tsdf, colorGrid)
        
    return tsdf
