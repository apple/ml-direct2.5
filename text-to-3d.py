#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import cv2
import sys
import time
import yaml
import torch
import xatlas
import random
import open3d as o3d
import imageio
import trimesh
import argparse
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler
from tqdm import tqdm
from kornia.losses import ssim_loss, total_variation
from RealESRGAN import RealESRGAN
import pymeshlab
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss


from diffusion.mvnormal_pipeline import MVDiffusionPipeline
from diffusion.mvunet import MVUNet2DConditionModel
from diffusion.condmvdiffusion_pipeline import CondMVDiffusionPipeline
from diffusion.cond_unet import CondMVUNet2DConditionModel
from meshing.remesh import calc_vertex_normals
from meshing.opt import MeshOptimizer
from util import camera_utils
from util.func import make_sphere, write_obj_with_texture, poisson_remesh, clean_mesh
from util.render import NormalsRenderer


class SoftClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clip(min=min, max=max)

    @staticmethod
    def backward(ctx, dL_dout):
        # identity
        return dL_dout.clone(), None, None

def create_renderers(model_paras, mesh_opt_paras, iterative_camera_paras):
    normal_renderers, us_list, vs_list, in_regions_list, extrinsics_blender_list  = [], [], [], [], []
    texture_renderers, extrinsics_blender_tex_list = [], []
    norm_w2cs = []
    # load paras
    num_views = model_paras['num_views']
    space_carving_res = mesh_opt_paras['space_carving_res']
    norm_iters, tex_iters = mesh_opt_paras['norm_iters'], mesh_opt_paras['tex_iters']

    fov, azimuth_span = iterative_camera_paras['fov'], iterative_camera_paras['azimuth_span']
    # normal camera paras
    norm_camers, norm_size = iterative_camera_paras['normal'], model_paras['normal_res']
    azimuth_starts, elevations, radii = norm_camers['azimuth_starts'], norm_camers['elevations'], norm_camers['radius']
    for i in range(norm_iters):
        azimuth_start, elevation, radius = azimuth_starts[i], elevations[i], radii[i]
        # create cameras, mesh optimizers, volume projection uvs
        intrinsics, extrinsics_blender = camera_utils.get_camera(num_views, elevation, fov, radius, azimuth_start, azimuth_span, norm_size)
        extrinsics_blender = extrinsics_blender.numpy().reshape(4, 3, 4)
        extrinsics_cv = []
        for e in extrinsics_blender:
            e = camera_utils.convert_blender_extrinsics_to_opencv(e)
            e = np.concatenate([e.astype(np.float32), np.array([[0, 0, 0, 1]])], axis=0)[None, ...]
            extrinsics_cv.append(e)
        extrinsics_cv = np.concatenate(extrinsics_cv, axis=0)
        
        K = intrinsics[0].reshape(3, 3).cpu().numpy()
        proj = torch.from_numpy(camera_utils.camera_intrinsic_to_opengl_projection(K, w=norm_size, h=norm_size, n=0.01, f=5.)).to(pipe_device)
        mv = torch.cat([torch.from_numpy(extrinsics_blender), torch.Tensor([[0, 0, 0, 1]]).repeat(4, 1).unsqueeze(1)], axis=1).reshape(4, 4, 4).to(pipe_device)
        extrinsics_blender = torch.from_numpy(extrinsics_blender).to(device=pipe_device, dtype=torch.float16)
        renderer = NormalsRenderer(mv.float(), proj.float(), [norm_size, norm_size], device=pipe_device)
        us, vs, in_regions = camera_utils.build_volumes_projections(extrinsics_cv, K, resolution=space_carving_res, size=0.65)
        
        normal_renderers.append(renderer)
        us_list.append(us)
        vs_list.append(vs)
        in_regions_list.append(in_regions)
        extrinsics_blender_list.append(extrinsics_blender)
        norm_w2cs.append(torch.from_numpy(extrinsics_cv))
        
    # texture camera paras  
    tex_camers, tex_size = iterative_camera_paras['texture'], model_paras['texture_res']
    tex_azimuth_starts, tex_elevations, radii = tex_camers['azimuth_starts'], tex_camers['elevations'], tex_camers['radius']
    for i in range(tex_iters):
        azimuth_start, elevation, radius = tex_azimuth_starts[i], tex_elevations[i], radii[i]
        # create cameras, mesh optimizers, volume projection uvs
        intrinsics512, extrinsics_blender512 = camera_utils.get_camera(num_views, elevation, fov, radius, azimuth_start, azimuth_span, tex_size)
        extrinsics_blender512 = extrinsics_blender512.numpy().reshape(4, 3, 4)
        extrinsics_cv512 = []
        for e in extrinsics_blender512:
            e = camera_utils.convert_blender_extrinsics_to_opencv(e)
            e = np.concatenate([e.astype(np.float32), np.array([[0, 0, 0, 1]])], axis=0)[None, ...]
            extrinsics_cv512.append(e)
        extrinsics_cv512 = np.concatenate(extrinsics_cv512, axis=0)
        if i == 0:
            tex_init_w2c = torch.from_numpy(extrinsics_cv512)
        
        K512 = intrinsics512[0].reshape(3, 3).cpu().numpy()
        proj512 = torch.from_numpy(camera_utils.camera_intrinsic_to_opengl_projection(K512, w=tex_size, h=tex_size, n=0.01, f=5.)).to(pipe_device)
        mv512 = torch.cat([torch.from_numpy(extrinsics_blender512), torch.Tensor([[0, 0, 0, 1]]).repeat(4, 1).unsqueeze(1)], axis=1).reshape(4, 4, 4).to(pipe_device)
        extrinsics_blender512 = torch.from_numpy(extrinsics_blender512).to(device=pipe_device, dtype=torch.float16)
        renderer_tex = NormalsRenderer(mv512.float(), proj512.float(), [tex_size, tex_size], device=pipe_device)

        texture_renderers.append(renderer_tex)
        extrinsics_blender_tex_list.append(extrinsics_blender512)   
    
    return normal_renderers, texture_renderers, us_list, vs_list, in_regions_list, extrinsics_blender_list, extrinsics_blender_tex_list, tex_init_w2c, \
        intrinsics512[0].reshape(3, 3).float(), norm_w2cs, intrinsics[0].reshape(3, 3).float()


def create_360_view_renderers(paras, size=512):
    # create final visual renderer
    nframe, elevation, azimuth_start, azimuth_end, fov, radius = \
        paras['nframe'], paras['elevation'], paras['azimuth_start'], paras['azimuth_end'], paras['fov'], paras['radius']
    
    elevations = [0]
    renderers = []
    for elevation in elevations:
        azimuths = np.arange(azimuth_start, azimuth_end, azimuth_end/nframe)
        # create cameras, mesh optimizers, volume projection uvs
        intrinsics512, extrinsics_blender512 = camera_utils.get_camera(nframe, elevation, fov, radius, azimuth_start, azimuth_end, size)
        extrinsics_blender512 = extrinsics_blender512.numpy().reshape(nframe, 3, 4)
        extrinsics_cv512 = []
        for e in extrinsics_blender512:
            e = camera_utils.convert_blender_extrinsics_to_opencv(e)
            e = np.concatenate([e.astype(np.float32), np.array([[0, 0, 0, 1]])], axis=0)[None, ...]
            extrinsics_cv512.append(e)
        extrinsics_cv512 = np.concatenate(extrinsics_cv512, axis=0)
        
        K512 = intrinsics512[0].reshape(3, 3).cpu().numpy()
        proj512 = torch.from_numpy(camera_utils.camera_intrinsic_to_opengl_projection(K512, w=size, h=size, n=0.01, f=5.)).to(pipe_device)
        mv512 = torch.cat([torch.from_numpy(extrinsics_blender512), torch.Tensor([[0, 0, 0, 1]]).repeat(nframe, 1).unsqueeze(1)], axis=1).reshape(nframe, 4, 4).to(pipe_device)
        extrinsics_blender512 = torch.from_numpy(extrinsics_blender512).to(device=pipe_device, dtype=torch.float16)
        renderer = NormalsRenderer(mv512.float(), proj512.float(), [size, size], device=pipe_device)

        renderers.append(renderer)
        
    return renderers


def set_seed(seed):
    random.seed(str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(prompt_or_prompt_file):
    if prompt_or_prompt_file.endswith('.txt'):
        with open(prompt_or_prompt_file, 'r') as f:
            prompts = f.read().splitlines()
    else:
        prompts = [prompt_or_prompt_file]
        
    return prompts
        

def config_parser(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
        
    model_paras = config['model']
    mesh_opt_paras = config['mesh_opt']
    iterative_camera_paras = config['iterative_camera']

    return config, model_paras, mesh_opt_paras, iterative_camera_paras
            
    

# main forward
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default-256-iter.yaml')
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--seed', type=str, default='random')
    parser.add_argument('--label', type=str, default='default')
    parser.add_argument('--save_intermediate', type=int, default=1)
    parser.add_argument('--results_root', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--prompts', type=str, default='configs/example_prompts.txt')

    # load top-level configs and generation configs yaml
    args = parser.parse_args()
    config, num_repeats, seeds_type, save_intermediate, results_root, gpu, captions = \
        args.config, args.num_repeats, args.seed, args.save_intermediate, args.results_root, args.gpu, args.prompts
    config, model_paras, mesh_opt_paras, iterative_camera_paras = config_parser(config)
    captions = load_prompts(captions)
    
    # positive int32 range for random seed
    seed_range = [0, 2147483647]
    pipe_device = f"cuda:{args.gpu}"

    if model_paras['superresolution']:
        model_paras['texture_res'] *= 4
        sr_model = RealESRGAN(pipe_device, scale=4)
        sr_model.load_weights('weights/RealESRGAN_x4.pth', download=True)
        
    # load pre-trained models
    # text-to-normal model
    mvunet = MVUNet2DConditionModel.from_pretrained(model_paras['mvnorm_pretrain_path'],
                                                    subfolder="unet", 
                                                    torch_dtype=torch.float16)
    MVpipe = MVDiffusionPipeline.from_pretrained(model_paras['base_model'], unet=mvunet, torch_dtype=torch.float16)
    MVpipe.scheduler = DDIMScheduler.from_config(MVpipe.scheduler.config)
    MVpipe = MVpipe.to(pipe_device)
    # normal-conditioned texture generation model
    cond_mvunet = CondMVUNet2DConditionModel.from_pretrained(model_paras['mvtexture_pretrain_path'],
                                                             subfolder="unet", 
                                                             torch_dtype=torch.float16)
    Cond_MVpipe = CondMVDiffusionPipeline.from_pretrained(model_paras['base_model'], unet=cond_mvunet, torch_dtype=torch.float16)
    Cond_MVpipe.scheduler = DDIMScheduler.from_config(Cond_MVpipe.scheduler.config)
    Cond_MVpipe = Cond_MVpipe.to(pipe_device)
    if mesh_opt_paras['use_rendered_norm']:
        vae = Cond_MVpipe.vae
        vae.requires_grad_(False)
    
    # inference
    # name the subfolder 
    norm_task, norm_ckpt = model_paras['mvnorm_pretrain_path'].split('/')[-2:]
    cond_task, cond_ckpt = model_paras['mvtexture_pretrain_path'].split('/')[-2:]

    # init normal blobs
    if model_paras['init_size'] > 0:
        image_root = os.path.join('preprocess/normal_blob_init', f"scale_{model_paras['init_size']}")
        init_images = []
        for i in range(4):
            init_images.append(cv2.imread(os.path.join(image_root, f'{i}.png'), cv2.IMREAD_UNCHANGED)[..., ::-1][None, ...] / 65535.)
        init_images = torch.from_numpy(np.concatenate(init_images, axis=0)).cuda().float().to(pipe_device)
        init_images = init_images.permute(0, 3, 1, 2) * 2 - 1
        init_latents = MVpipe.vae.encode(init_images.to(torch.float16)).latent_dist.sample()
        init_latents = init_latents * MVpipe.vae.config.scaling_factor
    else:
        init_latents = None
    
    # load requried
    normal_renderers, texture_renderers, us_list, vs_list, in_regions_list, extrinsics_blender_list, extrinsics_blender_tex_list, tex_init_w2c, K512, norm_w2cs, K256 = \
        create_renderers(model_paras, mesh_opt_paras, iterative_camera_paras)
    full_renderers = create_360_view_renderers(config['render_camera'])
    norm_iters, tex_iters, tex_act = mesh_opt_paras['norm_iters'], mesh_opt_paras['tex_iters'], mesh_opt_paras['activate']
    l1_weight, ssim_weight, tv_weight = mesh_opt_paras['l1_weight'], mesh_opt_paras['ssim_weight'], mesh_opt_paras['tv_weight']
     
    if model_paras['superresolution']:
        model_paras['texture_res'] = 256   
    if args.label == 'default':
        label = f'[{norm_iters}-{tex_iters}]-{norm_task}-{norm_ckpt}-{cond_task}-{cond_ckpt}'
    else:
        label = args.label
    results_root = os.path.join(args.results_root, label)
    os.makedirs(results_root, exist_ok=True)
    print('writing results to', results_root)
    
    # loop in prompts                                              
    for caption in captions:
        # generate current seeds
        if seeds_type == 'random':
            seeds = np.random.randint(seed_range[0], seed_range[1], size=(num_repeats, 2)) 
        else:
            seeds = []
        for i in range(num_repeats):
            print('use seed', seeds[i][0], seeds[i][1])
            norm_seed, tex_seed = seeds[i]
            file_prefix = f'{caption}_{norm_seed}_{tex_seed}'.replace(' ', '_')
            if save_intermediate:
                intermediate_path = os.path.join(results_root, file_prefix)
                os.makedirs(intermediate_path, exist_ok=True)
                
            # iteratively update
            norm_strengths, tex_strengths = mesh_opt_paras['norm_strengths'], mesh_opt_paras['tex_strengths']
            space_carving, space_carving_res = mesh_opt_paras['space_carving'], mesh_opt_paras['space_carving_res']
            normal_steps, texture_steps = mesh_opt_paras['normal_steps'], mesh_opt_paras['texture_steps']
            render_type, texture_map_res = mesh_opt_paras['render_type'], mesh_opt_paras['texture_map_res']
            size, tex_size = model_paras['normal_res'], model_paras['texture_res']
            bg_type, negative_prompt = model_paras['texture_model_background'], model_paras['negative_prompt']            
            if bg_type == 'gray':
                hypara_model2_background = torch.Tensor([0, 0, 0])
            elif bg_type == 'blue':
                hypara_model2_background = torch.Tensor([0, 0, 1])
            elif bg_type == 'black':
                hypara_model2_background = torch.Tensor([-1, -1, -1])
            elif bg_type == 'random gray':
                gray = (np.random.random() - 0.5) * 2
                hypara_model2_background = torch.Tensor([gray, gray, gray])
            st_instance = time.time()
            ###################################### Geomtry Optimization ############################## 
            for iter in range(norm_iters):
                #### load required data
                renderer, us, vs, in_regions, extrinsics_blender = \
                    normal_renderers[iter], us_list[iter], vs_list[iter], in_regions_list[iter], extrinsics_blender_list[iter]
                
                #### step 1: generate 4-view normal maps from given text prompt
                st = time.time()
                prompt = caption + ', normal map'
                cross_attention_kwargs = {'num_view': model_paras['num_views'], 'attention_weights': None}
                set_seed(norm_seed)
                if iter == 0:
                    norm_images, norm_latents, _ = MVpipe(
                        prompt, 
                        negative_prompt=None,
                        height=256,
                        width=256,
                        camera_intrinsics=None,
                        camera_extrinsics=extrinsics_blender.reshape(4, 12),
                        cross_attention_kwargs=cross_attention_kwargs,
                        guidance_scale=7.5,
                        num_images_per_prompt=model_paras['num_views'],
                        output_type='np',
                        num_inference_steps=model_paras['infer_steps'],
                        latents=init_latents,
                        return_dict=False,
                        )
                else:
                    norm_images, norm_latents, _ = MVpipe(
                        prompt, 
                        negative_prompt=None,
                        image=rendered_norm,
                        strength=norm_strengths[iter],
                        mask=render_invalid_mask,
                        height=256,
                        width=256,
                        camera_intrinsics=None,
                        camera_extrinsics=extrinsics_blender.reshape(4, 12),
                        cross_attention_kwargs=cross_attention_kwargs,
                        guidance_scale=7.5,
                        num_images_per_prompt=model_paras['num_views'],
                        output_type='np',
                        num_inference_steps=model_paras['infer_steps'],
                        # latents=init_latents,
                        return_dict=False,
                        )

                if save_intermediate:
                    norm_images_uint16 = np.concatenate(list(norm_images), axis=1)
                    norm_images_uint16 = (norm_images_uint16 * 65535.).astype(np.uint16)
                    cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_norm_{iter}.png"), norm_images_uint16[..., ::-1])
                print(f'step 1: generate 4-view normal maps from given text prompt takes {time.time() - st} seconds.')
 
                #### step 2: optimize a mesh from 4-view normal maps
                st = time.time()
                valid = (norm_images > 0.05).sum(axis=-1) > 0
                alpha, invalid = valid, ~valid
                norm_images = (norm_images * 2 - 1).reshape(-1, 3).T
                norm_images = np.array([[0, 0, 1],
                                        [0, 1, 0],
                                        [-1, 0, 0]], np.float32).dot(norm_images)#.T.reshape(4, size, size, 3)
                norm_images = norm_images.T.reshape(4, size, size, 3)
                
                # normalized norm to 1
                norm_images /= np.linalg.norm(norm_images, axis=-1)[..., None]
                norm_images[invalid] = -1
                norm_images = norm_images * 0.5 + 0.5    # [0, 1]
                
                norm_images = np.concatenate([norm_images, alpha[..., None]], axis=-1)
                
                # space carving only at first iteration
                if iter == 0:
                    if space_carving:
                        st_0 = time.time()
                        vertices, faces = camera_utils.space_carving(
                            alpha, us, vs, in_regions, resolution=space_carving_res, erosion=0, dilation=0, img_size=size)
                        print('space carving takes time: ', time.time() - st_0)
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                        if save_intermediate:
                            _ = mesh.export(os.path.join(intermediate_path, f"{file_prefix}_space_carving.obj"))
                        # mesh = mesh.simplify_quadric_decimation(face_count=3000)
                        mesh = mesh.simplify_quadric_decimation(face_count=1500)
                        vertices, faces = mesh.vertices, mesh.faces
                        vertices = torch.from_numpy(vertices).to(pipe_device).float()
                        faces = torch.from_numpy(faces.copy()).to(pipe_device).long()
                    else:
                        vertices, faces = make_sphere(level=2, radius=.5)
                    
                # normal meshing optimization
                target_norms = torch.from_numpy(norm_images).to(pipe_device).float()
                opt = MeshOptimizer(vertices.detach(), faces.detach(), laplacian_weight=0.2)
                vertices = opt.vertices
                for i_ in tqdm(range(normal_steps)):
                    opt.zero_grad()
                    normals = calc_vertex_normals(vertices, faces)
                    images = renderer.render(vertices, normals, faces)
                    normal_loss = (images[..., :3] - target_norms[..., :3]).abs().mean()
                    alpha_loss = (images[..., 3:4] - target_norms[..., 3:4]).abs().mean()
                    # normal consistency
                    _mesh = Meshes(verts=[vertices], faces=[faces])
                    normal_consist_loss = mesh_normal_consistency(_mesh)
                    loss = normal_loss * 1. + alpha_loss * 1. + normal_consist_loss * 1e-1
                    loss.backward()
                    opt.step()
                    vertices, faces = opt.remesh()
                    
                vertices, faces = vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                mesh = clean_mesh(mesh, thresh=0.01)
                vertices = torch.from_numpy(mesh.vertices).to(pipe_device).float()
                faces = torch.from_numpy(mesh.faces.copy()).to(pipe_device).long()    
  
                print(f'step 2: optimize a mesh from 4-view normal maps takes {time.time() - st} seconds.')
                # if save_intermediate:
                #     vertices0, faces0 = vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
                #     mesh = trimesh.Trimesh(vertices=vertices0, faces=faces0, process=False)
                #     _ = mesh.export(os.path.join(intermediate_path, f"{file_prefix}_iter{iter}.obj"))
                
                if iter != norm_iters - 1:
                    # compute normal mask at here
                    xyz_homo = torch.cat((vertices, torch.ones(vertices.shape[0],1,device=vertices.device)),axis=-1).T
                    xyz_cam = torch.bmm(norm_w2cs[iter].to(vertices.device).to(vertices.dtype), xyz_homo[None, ...].repeat(4, 1, 1))[:, :3]
                    xyz_cam_depth = xyz_cam[:, 2]
                    xyz_img = torch.bmm(K256[None, ...].to(vertices.device).to(vertices.dtype).repeat(4, 1, 1), xyz_cam)
                    xyz_img = xyz_img[:, :2] / xyz_img[:, 2:3]
                    
                    depth_img = renderer.render_depth(vertices, faces)
                    sample_grid = (xyz_img / K256[0, 2]) - 1
                    sample_grid[:, 1] = -1 * sample_grid[:, 1]
                    # sample depth in the depth image and compare it with projected depth to decide whether valid
                    sampled_values = torch.nn.functional.grid_sample(
                        depth_img.permute(0, 3, 1, 2), 
                        sample_grid.reshape(4, 2, -1, 1).permute(0, 2, 3, 1), 
                        mode='bilinear',
                        align_corners=False).permute(0, 2, 3, 1)
                    depth_sampled = sampled_values[:, :, 0, 0]
                    
                    for i in range(4):
                        diff = xyz_cam_depth[i] - depth_sampled[i]
                        if i == 0:
                            valid = diff < 0.02
                        else:
                            valid = torch.logical_or(diff < 0.02, valid)
                    invalid_vert_mask = ~valid 
                    invalid_vert_indices = torch.arange(vertices.shape[0]).to(vertices).long()[invalid_vert_mask]
                    invalid_face_mask = torch.isin(faces, invalid_vert_indices).sum(dim=-1) > 0
                    invalid_face_indices = torch.arange(faces.shape[0]).to(faces).long()[invalid_face_mask] + 1
                    
                    # render normal at next view
                    normals = calc_vertex_normals(vertices, faces)
                    rendered, rast_out = normal_renderers[iter + 1].render(vertices, normals, faces, return_triangle=True)
                    # compute mask
                    proj_tri = rast_out[..., -1] 
                    render_invalid_mask = torch.isin(proj_tri, invalid_face_indices).float()
                    if save_intermediate:
                        render_invalid_mask_np = render_invalid_mask.detach().cpu().numpy() * 255.
                        render_invalid_mask_uint8 = np.concatenate(list(render_invalid_mask_np), axis=1)
                        cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_norm_invalid_mask_{iter+1}.png"), render_invalid_mask_uint8)
                    
                    rendered_norm, rendered_alpha = rendered[..., :3], rendered[..., 3]
                    rendered_norm = rendered_norm.reshape(-1, 3).T * 2 - 1
                    rendered_alpha = rendered_alpha.reshape(-1)
                    # convert to blender coords
                    rendered_norm = torch.tensor([[0,  0, -1],
                                                  [0,  1,  0],
                                                  [1,  0,  0]]).to(rendered_norm) @ rendered_norm
                    rendered_norm = rendered_norm.T
                    rendered_norm[rendered_alpha == 0] = -1
                    rendered_norm = rendered_norm.reshape(4, size, size, 3)
                    if save_intermediate:
                        rendered_norm_np = rendered_norm.detach().cpu().numpy() * 0.5 + 0.5
                        rendered_norm_uint16 = np.concatenate(list(rendered_norm_np), axis=1)
                        rendered_norm_uint16 = (rendered_norm_uint16 * 65535.).astype(np.uint16)
                        cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_norm_init_{iter+1}.png"), rendered_norm_uint16[..., [2,1,0]])   
                    rendered_norm = rendered_norm.permute(0, 3, 1, 2)
                    # norm_latents = vae.encode(rendered_norm.to(torch.float16)).latent_dist.sample()
                    # norm_latents = norm_latents * vae.config.scaling_factor


            ###################################### Mesh parameterization ############################## 
            # xatlas parameterization
            st_0 = time.time()
            vertices, faces = vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh = mesh.simplify_quadric_decimation(face_count = max(10000, mesh.faces.shape[0] // 4))
            vertices = mesh.vertices
            faces = mesh.faces

            if save_intermediate:
                    cpg_vert_fix = mesh.vertices.copy()
                    # cpg_vert_fix[:,0] *= -1
                    cpg_vert_fix[:,2] *= -1
                    cpg_mesh = trimesh.Trimesh(vertices=cpg_vert_fix, faces=mesh.faces[:,::-1], process=False)
                    _ = cpg_mesh.export(os.path.join(intermediate_path, f"{file_prefix}_simp.obj"))
                    # repair mesh
                    ms = pymeshlab.MeshSet()
                    ms.load_new_mesh(os.path.join(intermediate_path, f"{file_prefix}_simp.obj"))
                    ms.compute_selection_by_non_manifold_edges_per_face()
                    ms.meshing_remove_selected_vertices_and_faces()
                    ms.save_current_mesh(os.path.join(intermediate_path, f"{file_prefix}_repair.obj"))

            
            ###################################### Texture Optimization ############################## 
            vertices, faces = torch.from_numpy(vertices).to(pipe_device).float(), torch.from_numpy(faces.astype(np.int32)).to(pipe_device)
            normals = calc_vertex_normals(vertices, faces.long())
            for iter in range(tex_iters):
                ### load required data
                renderer_tex = texture_renderers[iter]
                #### step 3: generate 4-view texture maps from normal maps and prompt
                st = time.time()
                if mesh_opt_paras['use_rendered_norm']:
                    rendered512 = renderer.render(vertices,normals,faces)
                    rendered_norm512, rendered_alpha512 = rendered512[..., :3], rendered512[..., 3]
                    rendered_norm512 = rendered_norm512.reshape(-1, 3).T * 2 - 1
                    rendered_alpha512 = rendered_alpha512.reshape(-1)
                    # convert to blender coords
                    rendered_norm512 = torch.tensor([[0,  0, -1],
                                                     [0,  1,  0],
                                                     [1,  0,  0]]).to(rendered_norm512) @ rendered_norm512
                    rendered_norm512 = rendered_norm512.T
                    
                    if hypara_model2_background is not None:
                        rendered_norm512[rendered_alpha512 < 0.5] = hypara_model2_background.to(rendered_norm512)
                    else:
                        rendered_norm512[rendered_alpha512 < 0.5] = -1
                        # rendered_norm512[rendered_alpha512 == 0] = 0
                    rendered_norm512 = rendered_norm512.reshape(4, tex_size, tex_size, 3)
                    
                    if save_intermediate:
                        rendered_norm512_np = rendered_norm512.detach().cpu().numpy() * 0.5 + 0.5
                        rendered_norm512_uint16 = np.concatenate(list(rendered_norm512_np), axis=1)
                        rendered_norm512_uint16 = (rendered_norm512_uint16 * 65535.).astype(np.uint16)
                        cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_norm_rendered_{iter}.png"), rendered_norm512_uint16[..., [2,1,0]])
                    
                    rendered_norm = rendered_norm512.permute(0, 3, 1, 2)
                    norm_latents = vae.encode(rendered_norm.to(torch.float16)).latent_dist.sample()
                    norm_latents = norm_latents * vae.config.scaling_factor
                
                cross_attention_kwargs = {'num_view': model_paras['num_views'], 'attention_weights': None}
                set_seed(tex_seed)
                texture_prompt = caption + ', 3d asset'
                if iter == 0:
                    texture_images = Cond_MVpipe(#caption, 
                                                 texture_prompt,
                                                 height=tex_size,
                                                 width=tex_size,
                                                 cond_latents=norm_latents,
                                                 negative_prompt=negative_prompt,
                                                 cross_attention_kwargs=cross_attention_kwargs,
                                                 guidance_scale=7.5,
                                                #  attention_mask=attn_mask,
                                                 #  guidance_scale=10.0,
                                                 output_type='np',
                                                 num_images_per_prompt=4).images
                else:
                    # inpainting pipeline
                    _, _, hr, wr = rendered_rgb.shape
                    if hr > 256:
                        rendered_rgb = torch.nn.functional.interpolate(rendered_rgb, (256, 256), mode='bilinear', align_corners=False)
                        inpainted_mask = torch.nn.functional.interpolate(inpainted_mask.float().unsqueeze(1), (256, 256), mode='bilinear', align_corners=False)[:, 0]
                    texture_images = Cond_MVpipe(#caption,
                                                 texture_prompt,
                                                 image=rendered_rgb,
                                                 strength=tex_strengths[iter],
                                                 mask=inpainted_mask.float(),
                                                 height=tex_size,
                                                 width=tex_size,
                                                 cond_latents=norm_latents,
                                                 negative_prompt=negative_prompt,
                                                 cross_attention_kwargs=cross_attention_kwargs,
                                                 guidance_scale=7.5,
                                                 output_type='np',
                                                 num_images_per_prompt=4).images
                    
                print(f'step 3: generate 4-view texture maps from normal maps and prompt takes {time.time() - st} seconds.')
                
                if save_intermediate:
                    texture_images_uint8 = np.concatenate(list(texture_images), axis=1)
                    texture_images_uint8 = (texture_images_uint8 * 255.).astype(np.uint8)
                    cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_texture_iter{iter}.png"), texture_images_uint8[..., ::-1])
                
                if model_paras['superresolution']:
                    st_sr = time.time()
                    texture_images = np.concatenate(list(texture_images), axis=1)
                    texture_images = sr_model.predict((texture_images * 255).astype(np.uint8))
                    texture_images = np.array(texture_images)
                    h, w, _ = texture_images.shape
                    texture_images = np.concatenate([
                        texture_images[:, :w//4][None], 
                        texture_images[:, w//4:w//2][None],
                        texture_images[:, w//2:w//4*3][None],
                        texture_images[:, w//4*3:][None]], axis=0) / 255.
                    
                    print(f'super-resolution takes {time.time() - st_sr} seconds.')
                    if save_intermediate:
                        texture_images_uint8 = np.concatenate(list(texture_images), axis=1)
                        cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_texture_sr_iter{iter}.png"), texture_images_uint8[..., ::-1] * 255.)

                st = time.time()
                vertices, faces = mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)
                vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
                # xatlas.ChartOptions` and `xatlas.PackOptions`.
                print('parameterization takes time: ', time.time() - st)
        
                vertices, faces = torch.from_numpy(vertices[vmapping.astype(np.int32)]).to(pipe_device), torch.from_numpy(indices.astype(np.int32)).to(pipe_device)
                uvs = torch.from_numpy(uvs).to(pipe_device).float()
                normals = calc_vertex_normals(vertices, faces.long())
                uv_faces = faces

                st_tex = time.time()
                target_textures = torch.from_numpy(texture_images).to(pipe_device).float()
                if render_type == 'texture':
                    if iter == 0:
                        st_tex_init = time.time()
                        if mesh_opt_paras['init_texture']:  # FIXME not tested
                            # init texture map from generated images
                            texture_map, valid_inpaint, unseen_valid_region, uv_alpha = \
                                renderer_tex.sample_textures(vertices, faces, uvs, uv_faces, target_textures, tex_init_w2c, K512, texture_map_res, random=False) 
                            visibility_map = valid_inpaint.float().unsqueeze(-1).repeat(1, 1, 3)
                            if save_intermediate:
                                texture_map_init = texture_map.detach().cpu().numpy() * 255.
                                cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_texture_init.png"), texture_map_init[..., ::-1])
                                cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_texture_visibility.png"), visibility_map.detach().cpu().numpy() * 255.)
                            if tex_act == 'sigmoid':
                                texture_map = torch.log(texture_map) - torch.log(1 - texture_map)
                                texture_map = texture_map.clip(-1e8, 1e8)
                            texture_map = torch.nn.Parameter(texture_map)
                        else:
                            texture_map, valid_inpaint, unseen_valid_region, uv_alpha = \
                                renderer_tex.sample_textures(vertices, faces, uvs, faces, target_textures, tex_init_w2c, K512, texture_map_res) 
                            visibility_map = valid_inpaint.float().unsqueeze(-1).repeat(1, 1, 3)
                            texture_map = torch.nn.Parameter(torch.ones([texture_map_res, texture_map_res, 3], dtype=torch.float32).cuda() * 0.5)
                            if save_intermediate:
                                cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_texture_visibility.png"), visibility_map.detach().cpu().numpy() * 255.)
                            
                        print(f'texure init takses {time.time() - st_tex_init} seconds.')
                        

                    texture_map.requires_grad = True
                    opt = torch.optim.Adam([texture_map], lr=0.2)
                    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5, last_epoch=-1)
                    for i_ in tqdm(range(texture_steps)):
                        opt.zero_grad()
                        images = renderer_tex.render_rgb(vertices, faces, uvs, uv_faces, texture_map)
                        if tex_act == 'tanh':
                            rgb, alpha = torch.tanh(images[..., :3]) * 0.5 + 0.5, images[..., 3] > 0
                        elif tex_act == 'sigmoid':
                            rgb, alpha = torch.sigmoid(images[..., :3]), images[..., 3] > 0
                        elif tex_act =='softclip':
                            rgb, alpha = SoftClip.apply(images[..., :3], 0.0, 1.0), images[..., 3] > 0
                        loss_l1 = (rgb-target_textures)[alpha].abs().mean()
                        # loss_l2 = ((rgb-target_textures)[alpha] ** 2).mean()
                        loss_ssim = ssim_loss(rgb, target_textures, window_size=5, reduction='none')[alpha].mean()
                        loss_tv = total_variation(texture_map.permute(2,0,1), reduction="mean").mean()
                        loss = loss_l1 * l1_weight + loss_ssim * ssim_weight + loss_tv * tv_weight #/ (iter + 1)
                        loss.backward()
                        opt.step()
                        scheduler.step()
               
                elif render_type == 'vertices':
                    raise NotImplementedError('vertices color is not implemented now')
                print(f'step 4: texture the models takes {time.time() - st_tex} seconds.')           
                
                ### save uv texture results & inpainted mesh
                if save_intermediate:
                    if tex_act == 'tanh':
                        texture_map_np = (torch.tanh(texture_map) * 0.5 + 0.5).detach().cpu().numpy()
                    elif tex_act =='sigmoid':
                        texture_map_np = torch.sigmoid(texture_map).detach().cpu().numpy()
                    elif tex_act =='softclip':
                        texture_map_np = SoftClip.apply(texture_map, 0.0, 1.0).detach().cpu().numpy()
                    im = Image.fromarray((texture_map_np * 255).astype(np.uint8))
                    im.save(os.path.join(intermediate_path, f"{file_prefix}_uv_texture_{iter+1}.png"))

                if iter == 0 and tex_iters > 1:
                    background = torch.ones([4, 1024, 1024, 3], dtype=torch.float32).to(pipe_device).float() * 0.5
                
                # if not final rendering
                if iter != tex_iters - 1:
                    # render normal at next view
                    images = texture_renderers[iter+1].render_rgb(vertices, faces, uvs, uv_faces, texture_map)
                    visibility_images = texture_renderers[iter+1].render_rgb(vertices, faces, uvs, uv_faces, visibility_map)
                    if tex_act == 'tanh':
                        rendered_rgb, alpha = torch.tanh(images[..., :3]) * 0.5 + 0.5, images[..., 3] > 0
                    elif tex_act == 'sigmoid':
                        rendered_rgb, alpha = torch.sigmoid(images[..., :3]), images[..., 3] > 0
                    elif tex_act =='softclip':
                        rendered_rgb, alpha = SoftClip.apply(images[..., :3], 0.0, 1.0), images[..., 3] > 0
                    # background replace
                    rendered_rgb[~alpha] = background[~alpha]
                    # visibility 
                    rendered_vis, alpha_vis = visibility_images[..., :3], visibility_images[..., 3] > 0
                    rendered_vis[~alpha_vis] = 0
                    rendered_vis[rendered_vis > 0] = 1
                    inpainted_mask = torch.logical_and(alpha_vis, (rendered_vis==0)[..., 0])
                    # rendered_vis[alpha_vis] = 1
                    if save_intermediate:
                        alpha = alpha.cpu().numpy()
                        rendered_rgb_np = rendered_rgb.detach().cpu().numpy()
                        rendered_rgb_np[~alpha] = 0
                        rendered_rgb_uint8 = np.concatenate(list(rendered_rgb_np), axis=1)
                        rendered_rgb_uint8 = (rendered_rgb_uint8 * 255.).astype(np.uint8)
                        cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_tex_rendered_{iter+1}.png"), rendered_rgb_uint8[..., [2,1,0]])  
                        
                        alpha_vis = alpha_vis.detach().cpu().numpy()
                        rendered_vis_np = rendered_vis.detach().cpu().numpy()
                        rendered_vis_np[~alpha_vis] = 0
                        # rendered_vis_np[alpha_vis] = 1
                        rendered_vis_uint8 = np.concatenate(list(rendered_vis_np), axis=1)
                        rendered_vis_uint8 = (rendered_vis_uint8 * 255.).astype(np.uint8)
                        cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_tex_rendered_vis_{iter+1}.png"), rendered_vis_uint8[..., [2,1,0]])
                        
                        inpainted_mask_vis_np = inpainted_mask.detach().cpu().numpy()
                        inpainted_mask_vis_uint8 = np.concatenate(list(inpainted_mask_vis_np), axis=1)
                        inpainted_mask_vis_uint8 = (inpainted_mask_vis_uint8 * 255.).astype(np.uint8)
                        cv2.imwrite(os.path.join(intermediate_path, f"{file_prefix}_tex_inpainted_mask_{iter+1}.png"), inpainted_mask_vis_uint8)
                    rendered_rgb = rendered_rgb.permute(0, 3, 1, 2) * 2 - 1

            ed_opti = time.time()
            print(f'finished optimization in {ed_opti - st_instance} seconds')
            
            # render videos
            with torch.no_grad():
                full_renderings0 = full_renderers[0].render_rgb(vertices, faces, uvs, uv_faces, texture_map)
                if tex_act == 'sigmoid':
                    full_RGB0, full_alpha0 = torch.sigmoid(full_renderings0[..., :3]), full_renderings0[..., 3]
                normals = calc_vertex_normals(vertices, faces.long())
                full_normals0 = full_renderers[0].render(vertices, normals, faces)[..., :3]
            frames = torch.cat([full_RGB0, full_normals0], axis=2)
            frames = list((frames.detach().cpu().numpy() * 255.).astype(np.uint8))
            render_file = os.path.join(results_root, f'{file_prefix}.mp4')
            imageio.mimsave(render_file, frames, fps=30)

            # save obj model
            uv = uvs.detach().cpu().numpy()
            uv[:, 1] = 1 - uv[:, 1]
            # flip y
            vertices[:, 1] = -vertices[:, 1]
            vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()#[:, ::-1]
            uv_faces = uv_faces.cpu().numpy()
            # to meshlab vis, z backward, x right, y up
            vertices = np.array([[1, 0,  0],
                                 [0, 0, -1],
                                 [0, 1,  0]], np.float32).dot(vertices.T).T
            
            filename = f'{results_root}/{file_prefix}.obj'
            # FIXME texture looks weird, but video is ok
            write_obj_with_texture(filename, vertices, faces, im, uv, uv_faces)
            ed_instance = time.time()
            print(f'finished tasks in {ed_instance - st_instance} seconds')
            
                
        
