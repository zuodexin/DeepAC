import os
import torch
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import cv2
import copy
import warnings
import pickle
import glob
from tqdm import tqdm
import ipdb
import sys
import pandas as pd

from ..utils.geometry.wrappers import Pose, Camera
from ..models import get_model
from ..utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight
from ..dataset.utils import read_image, resize, numpy_image_to_torch, crop, zero_pad, get_imgaug_seq
from ..utils.utils import project_correspondences_line, get_closest_template_view_index,\
    get_closest_k_template_view_index, get_bbox_from_p2d
from ..models.deep_ac import calculate_basic_line_data

# add bop_toolkit to the python path
sys.path.append('../PoseLab/bop_toolkit')
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout

def load_prerender(dp_split, dp_cad_model):
    obj_ids = dp_cad_model['obj_ids']
    prerender = {}
    print("Loading pre-rendered data...")
    for obj_id in tqdm(obj_ids):
        data_dir = os.path.dirname(dp_cad_model['model_tpath'].format(obj_id=obj_id))
        obj_name = f"obj_{obj_id:06d}"
        template_path = os.path.join(data_dir, obj_name, 'pre_render', f'{obj_name}.pkl')
        with open(template_path, "rb") as pkl_handle:
            pre_render_dict = pickle.load(pkl_handle)
        head = pre_render_dict['head']
        num_sample_contour_points = head['num_sample_contour_point']
        template_views = torch.from_numpy(pre_render_dict['template_view']).type(torch.float32)
        orientations = torch.from_numpy(pre_render_dict['orientation_in_body']).type(torch.float32)
        prerender[obj_name] = {
            'num_sample_contour_points': num_sample_contour_points,
            'template_views': template_views,
            'orientations': orientations
        }
    return prerender


def load_init_predictions(dp_split, prediction_path, detection_threshold=0, choose_obj=None):
    df = pd.read_csv(f"{prediction_path}")
    if detection_threshold > 0:
        df = df[df["score"] > detection_threshold]
    if choose_obj:
        df = df[df["obj_id"].isin(choose_obj)]
    scene_cameras = {}
    instances = []
    print("Loading initial poses...")
    for index, row in df.iterrows():
        scene_id = int(row["scene_id"])
        if scene_id not in scene_cameras:
            scene_cameras[scene_id] = inout.load_scene_camera(
                dp_split["scene_camera_tpath"].format(scene_id=scene_id)
            )
        im_id = int(row["im_id"])
        obj_id = int(row["obj_id"])
        score = float(row["score"])
        R = np.fromstring(row["R"], sep=" ").reshape(3, 3)
        t = np.fromstring(row["t"], sep=" ")
        pose0 = np.eye(4, dtype=np.float32)
        pose0[:3, :3] = R
        pose0[:3, 3] = t * 0.001
        K = scene_cameras[scene_id][im_id]["cam_K"]
        depth_scale = scene_cameras[scene_id][im_id]["depth_scale"]
        w, h = dp_split["im_size"]
        instance = dict(
            scene_id=scene_id,
            im_id=im_id,
            obj_id=obj_id,
            pose0=pose0,
            K=K,
            score=score,
            depth_scale=depth_scale
        )
        instances.append(instance)
    return instances

def preprocess_image(data_conf, img, bbox2d, camera):
    bbox2d[2:] += data_conf.crop_border * 2
    img, camera, bbox = crop(img, bbox2d, camera=camera, return_bbox=True)
    
    scales = (1, 1)
    if isinstance(data_conf.resize, int):
        if data_conf.resize_by == 'max':
            # print('img shape', img.shape)
            # print('img path', image_path)
            img, scales = resize(img, data_conf.resize, fn=max)
        elif (data_conf.resize_by == 'min' or (data_conf.resize_by == 'min_if' and min(*img.shape[:2]) < data_conf.resize)):
            img, scales = resize(img, data_conf.resize, fn=min)
    elif len(data_conf.resize) == 2:
        img, scales = resize(img, list(data_conf.resize))
    if scales != (1, 1):
        camera = camera.scale(scales)

    img, = zero_pad(data_conf.pad, img)
    img = img.astype(np.float32)
    return numpy_image_to_torch(img), camera

def refine_pose(cfg, model, dp_split, data_conf, pre_render_dict, init_predictions, video=None):
    
    refined_poses = []
    for idx, sample in enumerate(tqdm(init_predictions)):
        
        img_path = dp_split["rgb_tpath"].format(scene_id=sample["scene_id"], im_id=sample["im_id"])
        K = sample["K"].flatten()
        
        ori_image = read_image(img_path)
        height, width = ori_image.shape[:2]
        intrinsic_param = torch.tensor([width, height, K[0], K[4], K[2], K[5]], dtype=torch.float32)
        ori_camera = Camera(intrinsic_param)
        
        init_pose = sample["pose0"]
        init_R = init_pose[:3,:3]
        init_t = init_pose[:3,3]
        init_pose = Pose.from_Rt(init_R, init_t)
        prerender = pre_render_dict[f"obj_{sample['obj_id']:06d}"]
        orientations = prerender['orientations']
        template_views = prerender['template_views']
        num_sample_contour_points = prerender['num_sample_contour_points']
        
        
        indices = get_closest_k_template_view_index(init_pose, orientations,
                                                    data_conf.get_top_k_template_views * data_conf.skip_template_view)
        closest_template_views = torch.stack([template_views[ind * num_sample_contour_points:(ind + 1) * num_sample_contour_points, :]
                                                for ind in indices[::data_conf.skip_template_view]])
        closest_orientations_in_body = orientations[indices[::data_conf.skip_template_view]]
        data_lines = project_correspondences_line(closest_template_views[0], init_pose, ori_camera)
        bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])
        img, camera = preprocess_image(data_conf, ori_image, bbox2d.numpy().copy(), ori_camera)
        
        
        _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
                calculate_basic_line_data(closest_template_views[None][:, 0], init_pose[None]._data, camera[None]._data, 1, 0)
        total_fore_hist, total_back_hist = \
            model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image, 
                                                foreground_distance, background_distance, True)

        data = {
            'image': img[None].cuda(), # (3, 320, 320)
            'camera': camera[None].cuda(), # (1), src_open.utils.geometry.wrappers.Camera
            'body2view_pose': init_pose[None].cuda(), # (1), src_open.utils.geometry.wrappers.Pose
            'closest_template_views': closest_template_views[None].cuda(), # (5, 200, 8)
            'closest_orientations_in_body': closest_orientations_in_body[None].cuda(), # (5, 3)
            'fore_hist': total_fore_hist.cuda(), # (1, 32768)
            'back_hist': total_back_hist.cuda()  # (1, 32768)
        }
        pred = model._forward(data, visualize=False, tracking=True)
        refined_pose = pred['opt_body2view_pose'][-1][0].cpu()
        refined_poses.append(refined_pose.matrix_3x4.numpy())
        
        if cfg.output_video:
            pred['optimizing_result_imgs'] = []
            model.visualize_optimization(pred['opt_body2view_pose'][-1], pred)
            video.write(cv2.resize(pred['optimizing_result_imgs'][0][0], cfg.output_size))
            
    return refined_poses


def save_refined_poses(pred_path, init_predictions, refined_poses):
    results = []
    for idx, _ in enumerate(tqdm(refined_poses)):
        sample = init_predictions[idx]
        TCO = np.eye(4, dtype=np.float32)
        TCO[:3, :4] = refined_poses[idx]
        TCO[:3, 3] *= 1000
        result = dict(
            scene_id=f'{sample["scene_id"]}',
            im_id=f'{sample["im_id"]}',
            obj_id=f'{sample["obj_id"]}',
            score=f'{sample["score"]:.4f}',
            R=" ".join([f"{r:.4f}" for r in TCO[:3, :3].reshape(-1).tolist()]),
            t=" ".join([f"{tt:.4f}" for tt in TCO[:3, 3].reshape(-1).tolist()]),
            time="-1",
        )
        results.append(result)
    keys = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    with open(pred_path, "w") as fp:
        fp.write(f"{','.join(keys)}\n")
        for result in results:
            line_items = []
            for k in keys:
                line_items.append(f"{result[k]}")
            fp.write(f"{','.join(line_items)}\n")
    print("predictions saved to ", pred_path)

@torch.no_grad()
def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    logger = MyLightningLogger('DeepAC', cfg.save_dir)
    logger.dump_cfg(cfg, 'demo_cfg.yml')
    assert ('load_cfg' in cfg)
    # assert ('load_model' in cfg)
    assert (Path(cfg.load_cfg).exists())
    # assert (Path(cfg.load_model).exists())
    train_cfg = OmegaConf.load(cfg.load_cfg)
    data_conf = train_cfg.data
    logger.dump_cfg(train_cfg, 'train_cfg.yml')
    
    
    model = get_model(train_cfg.models.name)(train_cfg.models)
    ckpt = torch.load(cfg.load_model, map_location='cpu')
    if "pytorch-lightning_version" not in ckpt:
        warnings.warn(
            "Warning! Old .pth checkpoint is deprecated. "
            "Convert the checkpoint with tools/convert_old_checkpoint.py "
        )
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    logger.info("Loaded model weight from {}".format(cfg.load_model))
    model.cuda()
    model.eval()
    
    video = None
    if cfg.output_video:
        video = cv2.VideoWriter(os.path.join(logger.log_dir, "refine.mp4"),  # 
                                cv2.VideoWriter_fourcc(*'mp4v'), 30, cfg.output_size)
    
    dp_split = dataset_params.get_split_params(
        cfg.bop_datasets_path, cfg.dataset, cfg.split, cfg.split_type
    )
    dp_cad_model = dataset_params.get_model_params(cfg.bop_datasets_path,cfg.dataset, "cad")
    
    pre_render_dict = load_prerender(dp_split, dp_cad_model)    
    init_predictions = load_init_predictions(dp_split, cfg.prediction_path, detection_threshold=cfg.detection_threshold, choose_obj=cfg.choose_obj)
    
    logger.info("Refining poses...")
    refined_poses =  refine_pose(cfg, model, dp_split, data_conf, pre_render_dict, init_predictions, video=video)
    
    save_path = f"{logger.log_dir}/deepac_{os.path.basename(cfg.prediction_path)}"
    save_refined_poses(save_path, init_predictions, refined_poses)
    
    if cfg.output_video:
        video.release()
        logger.info("Video saved at {}".format(os.path.join(logger.log_dir, "refine.mp4")))
    