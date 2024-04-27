#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from matplotlib import image
import nvdiffrast.torch as dr
import torch

def _warmup(glctx, device):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        # return torch.tensor(*args, device='cuda', **kwargs)
        return torch.tensor(*args, device=device, **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class NormalsRenderer:
    
    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: tuple[int,int],
            device: str = 'cuda',
            ):
        self._mvp = proj @ mv #C,4,4
        self._image_size = image_size
        # self._glctx = dr.RasterizeGLContext()
        self._glctx = dr.RasterizeGLContext(device=device)
        _warmup(self._glctx, device)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            return_triangle=False,
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vert_col = (normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        if return_triangle:
            return col, rast_out
        else:
            return col #C,H,W,4
    
    def render_vert_color(self,
            vertices: torch.Tensor, #V,3 float
            vert_color: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        col,_ = dr.interpolate(vert_color, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

    def render_depth(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vertex_depth = vertices_clip[:, :, 2:3].contiguous()
        # vertex_depth = vertex_depth.repeat(1, 1, 3)   # to 3 channel
        col,_ = dr.interpolate(vertex_depth, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4
    
    
    def render_position(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            bias: float=0,
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        col,_ = dr.interpolate(vertices.contiguous() + bias, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

    def render_alpha(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,1

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vertex_depth = vertices_clip[:, :, 2:3].contiguous()
        col,_ = dr.interpolate(vertex_depth, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        #col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(alpha, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

    def render_rgb(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            uv_coords: torch.Tensor,
            uv_idxs: torch.Tensor,
            texture_map: torch.Tensor
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out, rast_out_db = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        
        texc_uv, texd_uv = dr.interpolate(uv_coords[None, ...], rast_out, uv_idxs, rast_db=rast_out_db, diff_attrs='all')
        texture_map_dr = dr.texture(texture_map[None, ...], texc_uv, texd_uv, filter_mode='linear-mipmap-linear', max_mip_level=4)
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1

        texture_map_dr = torch.where(alpha > 0, texture_map_dr, torch.tensor(0.0).cuda())

        texture_map_dr = torch.concat((texture_map_dr,alpha),dim=-1) #C,H,W,4
        texture_map_dr = dr.antialias(texture_map_dr, rast_out, vertices_clip, faces) #C,H,W,4
        return texture_map_dr #C,H,W,4

    def sample_textures(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            uv_coords: torch.Tensor,
            uv_idxs: torch.Tensor,
            # texture_map: torch.Tensor,
            rendered_images: torch.Tensor,
            world2camera: torch.Tensor,
            K: torch.Tensor,
            texture_size=1024,
            random=False,
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        
        input_xyz = torch.cat((uv_coords * 2 - 1.0, torch.ones(V,1,device=vertices.device) * 0.5, torch.ones(V,1,device=vertices.device)),axis=-1)
        rast_out, rast_out_db = dr.rasterize(self._glctx, input_xyz[None, ...], faces, resolution=[texture_size, texture_size], grad_db=False) #C,H,W,4
        
        # compute xyz coords of texture map pixels
        uv_alpha = (torch.clamp(rast_out[..., -1:], max=1) > 0).reshape(-1) #C,H,W,1
        uv_bary_coords, uv_triangles = rast_out[0, ..., :2].reshape(-1, 2), rast_out[0, ..., -1].reshape(-1).long()
        uv_triangles[uv_alpha] = uv_triangles[uv_alpha] - 1
        uv_faces = faces[uv_triangles]
        
        uv_xyz = uv_bary_coords[:, 0:1] * vertices[uv_faces[:, 0]] + \
            uv_bary_coords[:, 1:2] * vertices[uv_faces[:, 1]] + \
            (1 - uv_bary_coords.sum(-1))[:, None] * vertices[uv_faces[:, 2]]
        uv_xyz[~uv_alpha] = 0
        uv_xyz_homo = torch.cat((uv_xyz, torch.ones(uv_xyz.shape[0],1,device=vertices.device)),axis=-1).T
        
        # project uv xyz image to all camera views and get coords
        uv_xyz_cam = torch.bmm(world2camera.to(vertices.device).to(vertices.dtype), uv_xyz_homo[None, ...].repeat(4, 1, 1))[:, :3]
        uv_xyz_cam_depth = uv_xyz_cam[:, 2]
        uv_xyz_img = torch.bmm(K[None, ...].to(vertices.device).to(vertices.dtype).repeat(4, 1, 1), uv_xyz_cam)
        uv_xyz_img = uv_xyz_img[:, :2] / uv_xyz_img[:, 2:3]
        
        # render ground truth depth image
        depth_img = self.render_depth(vertices, faces)
        combined_RGBD = torch.cat((rendered_images, depth_img), dim=-1)
        sample_grid = (uv_xyz_img / K[0, 2]) - 1
        sample_grid[:, 1] = -1 * sample_grid[:, 1]
        # sample depth in the depth image and compare it with projected depth to decide whether valid
        sampled_values = torch.nn.functional.grid_sample(
            combined_RGBD.permute(0, 3, 1, 2), 
            sample_grid.reshape(4, 2, texture_size, texture_size).permute(0, 2, 3, 1), 
            mode='bilinear',
            align_corners=False).permute(0, 2, 3, 1)
        RGB_sampled = sampled_values[:, :, :, :3]
        depth_sampled = sampled_values[:, :, :, 3]
            
        # depth test
        # textures = torch.zeros(4, texture_size * texture_size, 3, device=vertices.device)
        if not random:
            textures = torch.ones(texture_size * texture_size, 3, device=vertices.device) * 0.5
        else:
            textures = torch.rand(texture_size * texture_size, 3, device=vertices.device) * 0.5
        valids = []
        for i in range(4):
            diff = uv_xyz_cam_depth[i] - depth_sampled[i].reshape(-1)
            valid = torch.logical_and(diff < 0.02, uv_alpha)
            valids.append(valid[None, ...])
            # textures[i][valid] = RGB_sampled[i].reshape(-1, 3)[valid]
            textures[valid] = RGB_sampled[i].reshape(-1, 3)[valid]

            # RGB_sampled[i].reshape(-1, 3)[~valid] = 0
            
            
        valids_count = torch.cat(valids, dim=0).sum(dim=0).float()
        inpaint_valid = valids_count > 0
        textures = textures.reshape(texture_size, texture_size, 3)
        unseen_valid_region = torch.logical_and(~inpaint_valid, uv_alpha).reshape(texture_size, texture_size)
        
        return textures, inpaint_valid.reshape(texture_size, texture_size), unseen_valid_region, uv_alpha.reshape(texture_size, texture_size)


def interpolate_face_attributes_python(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    F, FV, D = face_attributes.shape
    N, H, W, K, _ = barycentric_coords.shape

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = pix_to_face < 0
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (barycentric_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    return pixel_vals