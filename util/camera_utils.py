#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import math
import time
from skimage import measure
from scipy import ndimage

def fov_to_focal(fov, size):
    # convert fov angle in degree to focal
    return size / np.tan(fov * np.pi / 180.0 / 2.0) / 2.0


def focal_to_fov(focal, size):
    # convert focal to fov angle in degree
    return 2.0 * np.arctan(size / (2.0 * focal)) * 180.0 / np.pi


def set_camera_intrinsic(fov=60., size=512):
    focal = fov_to_focal(fov=fov, size=size)
    K = np.array([[focal,  0,  size / 2], 
                  [0,  focal,  size / 2],
                  [0,      0,         1]])
    return K


def set_blender_camera_extrinsics(azimuth, elevation, relative_radius=2., fov=60.):
    # compute absolute camera distance
    distance = relative_radius * 0.5 * (1 / np.tan(np.deg2rad(fov) / 2.0))
    
    # build blender camera to world matrix
    azimuth, elevation = np.deg2rad(azimuth), np.deg2rad(elevation)
    position = (
        distance * math.cos(azimuth) * math.cos(elevation),
        distance * math.sin(azimuth) * math.cos(elevation),
        distance * math.sin(elevation), 
    )
    
    # lookat camera in blender coordinates
    center = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    lookat = center - position
    lookat /= np.linalg.norm(lookat)
    right = np.cross(lookat, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, lookat)
    up /= np.linalg.norm(up)
    
    # blender world to camera matrix
    R_blender = np.array([right, up, -lookat])
    T_blender = -R_blender @ np.array(position)
    extrinsic_bcam = np.concatenate([R_blender, T_blender[:, None]], axis=-1)
    
    return extrinsic_bcam
    

def convert_blender_extrinsics_to_opencv(extrinsic_bcam):
    R_bcam2cv = np.array([[1, 0,  0], 
                          [0, -1, 0],
                          [0, 0, -1]], np.float32)
    R, t = R_bcam2cv @ extrinsic_bcam[:3, :3], R_bcam2cv @ extrinsic_bcam[:3, 3]
    extrinsic_cv = np.concatenate([R, t[..., None]], axis=1)
    
    return extrinsic_cv


def camera_intrinsic_to_opengl_projection(intrinsic,w=512,h=512,n=0,f=5,flip_y=False):
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    
    proj = np.array([
        [2.*fx/w,   0,    1-2*cx/w,           0],
        [0,    2*fy/h,   -1+2*cy/h,           0],
        [0,         0,(-f-n)/(f-n),-2*f*n/(f-n)],
        [0,         0,          -1,           0]
        ])
        
    if flip_y:
        proj[1,:] *= -1

    return proj


def generate_camera_parameters(azimuth, elevation=15., fov=60., radius=2., size=512):
    K = set_camera_intrinsic(fov, size)
    RT = set_blender_camera_extrinsics(azimuth, elevation, radius, fov)
    
    return K, RT


def get_camera(num_views, elevation=15, fov=60., radius=2., azimuth_start=0, azimuth_span=360, size=256):
    angle_gap = azimuth_span / num_views
    intrinsics, extrinsics = [], []
    for azimuth in np.arange(azimuth_start, azimuth_span+azimuth_start, angle_gap):
        K, RT = generate_camera_parameters(azimuth, elevation, fov, radius, size)
        intrinsics.append(torch.from_numpy(K.reshape(1, -1)))
        extrinsics.append(torch.from_numpy(RT.reshape(1, -1)))
    intrinsics = torch.cat(intrinsics, dim=0)
    extrinsics = torch.cat(extrinsics, dim=0)
    
    return intrinsics, extrinsics


def build_volumes_projections(extrinsics, intrinsic, resolution=128, size=0.5):
    # build volumes in [-size, size], and compute projections on each images
    x_coords, y_coords, z_coords = np.meshgrid(np.arange(resolution), np.arange(resolution), np.arange(resolution))
    volume_coords = np.stack([y_coords, x_coords, z_coords], axis=-1).reshape(-1, 3).T   # [3, res^3]
    volume_coords = (volume_coords + 0.5) / resolution * 2 * size - size
    num_cams = extrinsics.shape[0]
    us, vs, in_regions = [], [], []
    for i in range(num_cams):
        # project to image coords
        ext = extrinsics[i]
        R, T = ext[:3, :3], ext[:3, 3]
        proj_coords = intrinsic @ (R @ volume_coords + T[:, None])
        proj_coords = proj_coords[:2] / proj_coords[2]
        u, v = np.round(proj_coords[0]).astype(np.int32), np.round(proj_coords[1]).astype(np.int32)
        in_region = np.logical_and(
            np.logical_and(u > 0, u < 256 - 1),
            np.logical_and(v > 0, v < 256 - 1))
        us.append(u[None])
        vs.append(v[None])
        in_regions.append(in_region[None])
    us = np.concatenate(us, axis=0)
    vs = np.concatenate(vs, axis=0)
    in_regions = np.concatenate(in_regions, axis=0)
    
    return us, vs, in_regions


def space_carving(alphas, us, vs, in_regions, resolution, erosion=0, dilation=0, img_size=256):
    num_cams = alphas.shape[0]
    valids = np.zeros([num_cams, resolution * resolution * resolution], np.float32)
    # valids = np.ones([num_cams, resolution * resolution * resolution], np.float32)
    for i in range(num_cams):
        alpha, u, v, in_region = alphas[i], us[i], vs[i], in_regions[i]
        values = alpha[img_size - 1 - v[in_region], u[in_region]]
        valids[i][~in_region] = 0
        valids[i][in_region] = values
    valids = valids.sum(axis=0) == num_cams
    valids = valids.reshape(resolution, resolution, resolution)
    # dilation for loose supervision
    if dilation > 0:
        valids = ndimage.binary_dilation(valids, iterations=dilation)
    if erosion > 0:
        valids = ndimage.binary_erosion(valids, iterations=erosion)

    verts, faces, normals, values = measure.marching_cubes(valids, 0.5, gradient_direction='ascent')
    verts = verts / (resolution - 1.) - 0.5
    
    return verts, faces
