import numpy as np
import ultralytics.engine.results
from argparse import ArgumentParser
import torchvision

from gaussian_model import GaussianModel
from arguments import OptimizationParams
from datasets import *
from utils.utils import *
from utils.mapper_utils import *


class Mapper:

    def __init__(self, config: dict, dataset: BaseDataset) -> None:

        self.config = config
        self.alpha_thre = config['alpha_thre']
        self.uniform_seed_interval = config['uniform_seed_interval']
        self.dataset = dataset
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

    def new(self, frame_id: int, c2w: np.ndarray, yolo_result: ultralytics.engine.results.Results,
            object_idx: int) -> GaussianModel:

        gs = GaussianModel(0)
        gs.training_setup(self.opt)
        gs = self.update(frame_id, c2w, yolo_result, gs, object_idx, is_new=True)

        return gs

    def update(self, frame_id: int, c2w: np.ndarray, yolo_result: ultralytics.engine.results.Results,
               submap: GaussianModel, object_idx: int, is_new=False) -> GaussianModel:

        _, gt_color, gt_depth, _ = self.dataset[frame_id]
        w2c = np.linalg.inv(c2w)
        obj_mask = torch.squeeze(yolo_result.masks[object_idx].data).cpu().numpy()
        masked_color = (gt_color.transpose((2, 0, 1)) * obj_mask).transpose((1, 2, 0))
        masked_depth = gt_depth * obj_mask  # non-object area have zero depth
        keyframe = {
            "color": masked_color,
            "depth": masked_depth,
            "mask": obj_mask,
            "c2w": c2w,
            "render_settings": get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, w2c)}

        seeding_mask = self.compute_seeding_mask(submap, keyframe, is_new)
        pts = self.seed_new_points(keyframe, seeding_mask, self.dataset.intrinsics, is_new)
        new_pts_num = self.grow_submap(c2w, submap, pts)
        # print("New points num: %d" % new_pts_num)
        # @TODO: optimize submap

        return submap

    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, is_new: bool) -> np.ndarray:

        if is_new:
            color_for_mask = (keyframe["color"] * 255).astype(np.uint8)
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        else:
            render_dict = render_gs([gaussian_model], keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)
            gt_depth_tensor = np2torch(keyframe["depth"], device='cuda')[None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch2np(seeding_mask[0])

        return seeding_mask

    def seed_new_points(self, keyframe: dict, seeding_mask: np.ndarray,
                        intrinsics: np.ndarray, is_new: bool) -> np.ndarray:

        pts = create_point_cloud(keyframe["color"], 1.005 * keyframe["depth"], intrinsics, keyframe["c2w"])
        flat_gt_depth = keyframe["depth"].flatten()
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in masked_depth
        # pts = pts[non_zero_depth_mask]
        valid_ids = np.flatnonzero(seeding_mask)

        if is_new:
            if self.uniform_seed_interval < 0:
                uniform_ids = np.arange(pts.shape[0])
            else:
                num = np.int32(pts.shape[0] / self.uniform_seed_interval)
                uniform_ids = np.random.choice(pts.shape[0], num, replace=False)    # sample points uniformly
            combined_ids = np.concatenate([uniform_ids, valid_ids])
            sample_ids = np.unique(combined_ids)
        else:
            if self.uniform_seed_interval < 0:
                sample_ids = valid_ids
            else:
                num = np.int32(pts.shape[0] / self.uniform_seed_interval)
                sample_ids = np.random.choice(valid_ids, num, replace=False)
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]

        return pts[sample_ids, :].astype(np.float32)

    def grow_submap(self, c2w: np.ndarray, submap: GaussianModel, pts: np.ndarray) -> int:

        # @TODO: filter the points
        new_pts_ids = np.arange(pts.shape[0])
        cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0)
        submap.add_points(cloud_to_add)
        submap._features_dc.requires_grad = False
        submap._features_rest.requires_grad = False
        # @TODO: Re-enable terminal output
        # print("Gaussian model size", submap.get_size())

        return new_pts_ids.shape[0]
