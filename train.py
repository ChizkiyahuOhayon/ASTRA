#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, get_flow_loss, get_depth_loss, get_anisotropy_loss, get_lidar_depth_loss, get_free_space_loss
from utils.depth_utils import get_scaled_shifted_depth
from utils.flow_utils import flow_to_img, get_img_flow
from gaussian_renderer import render, USE_ABS_RASTERIZER
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_config
from scene.env import EnvironmentMap
from scene.photometric import PhotometricSpline
from scene.cameras import Camera
from torch.utils.tensorboard import SummaryWriter

def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    assert (not opt.abs_grad) or USE_ABS_RASTERIZER, \
        "--abs_grad needs the absolute-gradient rasterizer: run with ADGS_RASTERIZER=abs"
    gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
    env_map = EnvironmentMap(**dataset.env_args)
    gaussians.mip_filter = bool(getattr(dataset, 'mip_filter', False))
    scene = Scene(dataset, gaussians, env_map)
    photo = scene.photo

    # LiDAR returns, keyed by frame, held in GPU memory: one scene is a few tens of
    # megabytes and the training loop must not touch the NAS.
    lidar_gt = {}
    if opt.lambda_lidar > 0.0 or opt.lambda_free > 0.0:
        lidar_dir = dataset.lidar_dir
        for view in scene.getTrainCameras():
            path = os.path.join(lidar_dir, "%06d.npz" % int(view.fid))
            if not os.path.exists(path):
                continue
            uvz = np.load(path)["lidar_uvz"]
            uv = torch.from_numpy(uvz[:, :2]).round().long().cuda()
            uv[:, 0].clamp_(0, view.image_width - 1)
            uv[:, 1].clamp_(0, view.image_height - 1)
            z = torch.from_numpy(uvz[:, 2]).cuda()
            lidar_gt[int(view.fid)] = (uv, 1.0 / z.clamp(min=1e-3), z)
        print("LiDAR supervision: %d/%d training frames" % (len(lidar_gt), len(scene.getTrainCameras())))

    gaussians.training_setup(opt)
    env_map.training_setup(opt)
    photo.training_setup(opt)
    gaussians.compute_3D_filter(scene.getTrainCameras())

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if opt.data_sample == 'order':
            viewpoint_cam: Camera = viewpoint_stack.pop(0)
        elif opt.data_sample == 'stack':
            choice = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cam: Camera = viewpoint_stack.pop(choice)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if opt.lambda_flow > 0.0 and viewpoint_cam.flow is not None:
            flow_choice = randint(0, len(viewpoint_cam.flow) - 1)
            flow_pkg = viewpoint_cam.flow[flow_choice]
            flow_pkg = [a.cuda() if torch.is_tensor(a) else a for a in flow_pkg]
        else:
            flow_pkg = None

        render_pkg = render(viewpoint_cam, gaussians, env_map, pipe, photo, flow_pkg=flow_pkg, render_objmask=opt.lambda_obj > 0.0)
        image, visibility_filter, radii = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        dssim_loss = (1.0 - ssim(image, gt_image))
        depth_loss, flow_loss, obj_loss, sky_loss, sigma_loss, reg_loss, reg_sigma_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        dyn_loss, aniso_loss, lidar_loss, free_loss = 0.0, 0.0, 0.0, 0.0

        if opt.lambda_depth > 0.0:
            assert pipe.inv_depth, 'Depth-Any-Thing V2 monocular depth supervision should only support 1/d.'
            gt_depth = viewpoint_cam.depth.cuda()
            # 'inv' = stock. 'nosky' isolates the mask, 'log' adds the log-space comparison,
            # so the two changes can be ablated apart.
            if opt.depth_space == 'inv':
                depth_loss = get_depth_loss(render_pkg['depth'], gt_depth)
            else:
                sky_mask = (viewpoint_cam.sky.cuda() < 0.5).float()
                depth_loss = get_depth_loss(render_pkg['depth'], gt_depth, mask=sky_mask,
                                            space='log' if opt.depth_space == 'log' else 'inv')

        if opt.lambda_flow > 0.0 and flow_pkg is not None:
            flow_loss = get_flow_loss(render_pkg['img_flow'], flow_pkg, render_pkg['img_opacity'], dist=gaussians.scene_extent * 1e-3)
        
        if opt.lambda_obj > 0.0:
            gt_semantic = viewpoint_cam.semantic.cuda()
            pred_semantic = torch.clip(render_pkg['img_semantic'], 1e-3, 1.0 - 1e-3)
            obj_loss = torch.nn.functional.binary_cross_entropy(pred_semantic[0], (gt_semantic > 0).float())

        if opt.lambda_dyn > 0.0 and opt.soft_dyn:
            # Occam: a Gaussian has to earn the right to move.
            dyn_loss = torch.mean(gaussians.get_dynamicity)

        if opt.lambda_sky > 0.0:
            gt_sky = viewpoint_cam.sky.cuda()
            pred_sky = torch.clip(render_pkg['img_opacity'], 1e-3, 1.0 - 1e-3)
            sky_loss = torch.nn.functional.binary_cross_entropy(1.0 - pred_sky, gt_sky)
        
        if int(viewpoint_cam.fid) in lidar_gt:
            uv, inv_z, z = lidar_gt[int(viewpoint_cam.fid)]
            if opt.lambda_lidar > 0.0:
                lidar_loss = get_lidar_depth_loss(
                    render_pkg['depth'], render_pkg['img_opacity'], uv, inv_z,
                    opt.lidar_free_weight, opt.lidar_quantile)
            if opt.lambda_free > 0.0:
                free_loss = get_free_space_loss(
                    render_pkg['xyz'], render_pkg['opacity'].squeeze(-1), viewpoint_cam,
                    uv, z, opt.free_margin, opt.free_min_depth)

        if opt.lambda_aniso > 0.0:
            aniso_loss = get_anisotropy_loss(
                gaussians.get_scaling, render_pkg['visibility_filter'], opt.aniso_tau)

        if opt.lambda_reg > 0.0:
            deform_param = gaussians.get_gated_xyz_deform_param[gaussians.obj_near_idx]  # P, K, 3, C
            reg_loss = torch.mean(torch.sum(torch.var(deform_param, dim=1), dim=-1))

        if opt.lambda_sigma > 0.0:
            time_sigma = torch.exp(gaussians.gs_time_sigma)
            sigma_loss = torch.mean(torch.abs(gaussians.frame_gap / torch.mean(time_sigma, dim=-1)))
            if opt.lambda_sigma_reg > 0.0:
                time_sigma = gaussians.gs_time_sigma[gaussians.obj_near_idx]  # P, K, 2
                reg_sigma_loss = torch.mean(torch.sum(torch.var(time_sigma, dim=1), dim=-1))

        loss = (1.0 - opt.lambda_dssim) * opt.lambda_l1 * Ll1 + opt.lambda_dssim * dssim_loss
        loss += depth_loss * opt.lambda_depth + flow_loss * opt.lambda_flow
        loss += sky_loss * opt.lambda_sky + obj_loss * opt.lambda_obj
        loss += sigma_loss * opt.lambda_sigma + reg_loss * opt.lambda_reg + reg_sigma_loss * opt.lambda_sigma_reg
        loss += dyn_loss * opt.lambda_dyn + aniso_loss * opt.lambda_aniso + lidar_loss * opt.lambda_lidar + free_loss * opt.lambda_free
        loss.backward()

        log_losses = {
            'total_loss': loss,
            'depth_loss': depth_loss,
            'l1_loss': Ll1,
            'dssim_loss': dssim_loss,
            'flow_loss': flow_loss,
            'obj_loss': obj_loss,
            'sky_loss': sky_loss,
            'sigma_loss': sigma_loss,
            'reg_loss': reg_loss,
            'dyn_loss': dyn_loss,
        }

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "pts": gaussians.get_pts_num,
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, log_losses, opt, testing_iterations, scene, render, (pipe,))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(render_pkg)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_scene_grad_threshold, opt.densify_obj_grad_threshold, 0.005, iteration > opt.opacity_reset_interval)
                    gaussians.compute_3D_filter(scene.getTrainCameras())
                elif gaussians.use_near_idx and iteration % opt.near_idx_reset_interval == 0:
                    gaussians.set_obj_near_idx()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                if gaussians.motion_optimizer is not None:
                    gaussians.motion_optimizer.step()
                    gaussians.motion_optimizer.zero_grad(set_to_none=True)
                env_map.optimizer.step()
                photo.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                env_map.optimizer.zero_grad(set_to_none=True)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer

def training_report(tb_writer, iteration, losses, opt, testing_iterations, scene : Scene, renderFunc, renderArgs):
    for l_name, l_value in losses.items():
        tb_writer.add_scalar('train_loss_patches/{}'.format(l_name), l_value, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if scene.gaussians.soft_dyn:
            w = scene.gaussians.get_dynamicity.detach()
            w_obj = scene.gaussians.get_obj_dyn.detach()
            print('\n[ITER {}] Dynamicity: mean {:.4f} | object branch: mean {:.4f}, frac>0.5 {:.4f}, frac<0.1 {:.4f}'.format(
                iteration, w.mean().item(), w_obj.mean().item(),
                (w_obj > 0.5).float().mean().item(), (w_obj < 0.1).float().mean().item()))
            tb_writer.add_scalar('dyn/obj_mean', w_obj.mean().item(), iteration)
            tb_writer.add_scalar('dyn/obj_frac_static', (w_obj < 0.1).float().mean().item(), iteration)
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if idx < 5:
                        if viewpoint.flow is not None:
                            gt_flow_pkg = viewpoint.flow[0]
                            gt_flow_pkg = [a.cuda() if torch.is_tensor(a) else a for a in gt_flow_pkg]
                            flow_time, K, R, T, gt_flow, gt_flow_vis = gt_flow_pkg
                        else:
                            gt_flow_pkg = None

                        render_pkg = renderFunc(viewpoint, scene.gaussians, scene.env_map, *renderArgs, photo=scene.photo, flow_pkg=gt_flow_pkg, render_objmask=True)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_depth = viewpoint.depth.to('cuda')
                        gt_obj = (viewpoint.semantic.to('cuda') > 0).float()
                        error_map = torch.abs((image - gt_image))
                        
                        background = render_pkg['background']
                        foreground = render_pkg['foreground']
                        obj_map = render_pkg['img_semantic'].repeat(3, 1, 1)

                        if opt.lambda_depth > 0.0:
                            depthmap = get_scaled_shifted_depth(render_pkg['depth'], gt_depth)
                        else:
                            depthmap = render_pkg['depth']
                            depthmap = (depthmap - torch.min(depthmap)) / (torch.max(depthmap) - torch.min(depthmap))
                        depthmap = torch.clamp(depthmap, 0.0, 1.0)[None, ...]
                        gt_depth = torch.clamp(gt_depth, 0.0, 1.0)[None, ...]
                        
                        # flow = flow_to_img(render_pkg['img_flow'], gt_flow_vis) if gt_flow_pkg is not None else None
                        flow = flow_to_img(get_img_flow(render_pkg['img_flow'], gt_flow_pkg, dist=scene.gaussians.scene_extent * 1e-3), gt_flow_vis) if gt_flow_pkg is not None else None

                        if iteration == min(testing_iterations):
                            tb_writer.add_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                            tb_writer.add_image(config['name'] + "_view_{}/depth_gt".format(viewpoint.image_name), gt_depth.repeat(3, 1, 1), global_step=iteration)
                            tb_writer.add_image(config['name'] + "_view_{}/sky_gt".format(viewpoint.image_name), viewpoint.sky[None].repeat(3, 1, 1), global_step=iteration)
                            tb_writer.add_image(config['name'] + "_view_{}/obj_gt".format(viewpoint.image_name), gt_obj[None].repeat(3, 1, 1), global_step=iteration)
                            if gt_flow_pkg is not None:
                                tb_writer.add_image(config['name'] + "_view_{}/flow_gt".format(viewpoint.image_name), flow_to_img(gt_flow, gt_flow_vis), global_step=iteration)
                            
                        tb_writer.add_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/opacity".format(viewpoint.image_name), render_pkg['img_opacity'].repeat(3, 1, 1), global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depthmap.repeat(3, 1, 1), global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/foreground".format(viewpoint.image_name), foreground, global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/background".format(viewpoint.image_name), background, global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/error_map".format(viewpoint.image_name), error_map, global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/obj".format(viewpoint.image_name), obj_map, global_step=iteration)
                        if flow is not None:
                            tb_writer.add_image(config['name'] + "_view_{}/flow".format(viewpoint.image_name), flow, global_step=iteration)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, scene.env_map, *renderArgs)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        error_map = torch.abs((image - gt_image))
                    
                    l1_test += error_map.mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('points/total_points', scene.gaussians.get_pts_num, iteration)
            tb_writer.add_scalar('points/scene_points', scene.gaussians.get_scene_pts_num, iteration)
            tb_writer.add_scalar('points/obj_points', scene.gaussians.get_obj_pts_num, iteration)

        print("\n[ITER {}] Points: Total {} Scene {} Object {}".format(iteration, scene.gaussians.get_pts_num, scene.gaussians.get_scene_pts_num, scene.gaussians.get_obj_pts_num))

        torch.cuda.empty_cache()

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', '-c', type=str, default=None)
    config_path = parser.parse_known_args()[0].config
    if config_path is not None:
        assert os.path.exists(config_path)
        print("Find Config:", config_path)
        config = get_config(config_path)
    else:
        config = None
    lp = ModelParams(parser, config)
    op = OptimizationParams(parser, config)
    pp = PipelineParams(parser, config)
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[60_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.test_iterations += list(range(10_000, args.iterations, 10_000))
    args.test_iterations.append(args.iterations)

    args.data_device = "cuda:0" if args.data_device == 'cuda' else args.data_device
    torch.cuda.set_device(args.data_device)
    
    if not args.quiet:
        print(vars(args))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
