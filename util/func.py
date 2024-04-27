#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import io
import torch
import numpy as np
from pathlib import Path
import trimesh
import imageio
import cv2
import open3d as o3d


def write_obj_with_texture(obj_name, vertices, triangles, texture, uv_coords, uv_triangles):
    ''' Save 3D face model with texture represented by texture map.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        texture: shape = (256,256,3)
        uv_coords: shape = (nver, 3) max value<=1
    '''
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    base_name = os.path.basename(obj_name)
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '_texture.png')
    
    triangles = triangles.copy()
    triangles += 1 # mesh lab start with 1
    uv_triangles = uv_triangles.copy()
    uv_triangles += 1
    
    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        s = "mtllib {}\n".format(os.path.basename(mtl_name))
        f.write(s)
        f.write("usemtl MeshTexture\n")
        # s = 'usemtl {}\n'.format(base_name.split('.')[0])
        # f.write(s)

        # write vertices
        for i in range(vertices.shape[0]):
            s = 'v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)
        
        # write uv coords
        for i in range(uv_coords.shape[0]):
            # s = 'vt {} {}\n'.format(uv_coords[i,0], 1 - uv_coords[i,1])
            s = 'vt {} {}\n'.format(uv_coords[i,0], uv_coords[i,1])
            f.write(s)

        

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[0]):
            s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i,2], uv_triangles[i,2], triangles[i,1], uv_triangles[i,1], triangles[i,0], uv_triangles[i,0])
            f.write(s)

    # write mtl
    with open(mtl_name, 'w') as f:
        f.write("newmtl MeshTexture\n")
        s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
        f.write(s)

    # write texture as png
    texture.save(texture_name)


def make_sphere(level:int=2,radius=1.,device='cuda') -> tuple[torch.Tensor,torch.Tensor]:
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=1.0, color=None)
    vertices = torch.tensor(sphere.vertices, device=device, dtype=torch.float32) * radius
    faces = torch.tensor(sphere.faces, device=device, dtype=torch.long)
    return vertices,faces


def poisson_remesh(mesh, depth=7):
    mesh_o3d = mesh.as_open3d
    mesh_o3d.compute_vertex_normals()
    sample_pcd = mesh_o3d.sample_points_uniformly(mesh.faces.shape[0]*5)
    remesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(sample_pcd, depth=depth, n_threads=4)
    mesh = trimesh.Trimesh(vertices=np.asarray(remesh_o3d.vertices), faces=np.asarray(remesh_o3d.triangles), process=False)
    return mesh


def clean_mesh(mesh, thresh=0.05):
    components = mesh.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float32)
    max_area = areas.max()
    filtered = [components[i] for i in range(len(components)) if areas[i] > max_area*thresh]
    mesh = sum(filtered[1:], start=filtered[0])
    return mesh
