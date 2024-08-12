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
        keyframe = {
            "color": gt_color,
            "depth": gt_depth,
            "mask": torch.squeeze(yolo_result.masks[object_idx].data).cpu().numpy(),
            "c2w": c2w,
            "render_settings": get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, w2c)}

        pts = self.seed_new_points(keyframe, self.dataset.intrinsics, is_new)
        new_pts_num = self.grow_submap(c2w, submap, pts)
        # @TODO: optimize submap

        return submap

    def seed_new_points(self, keyframe: dict, intrinsics: np.ndarray, is_new: bool) -> np.ndarray:

        obj_mask = keyframe["mask"]
        gt_color = keyframe["color"]
        gt_depth = keyframe["depth"]
        gt_color = (gt_color.transpose((2, 0, 1)) * obj_mask).transpose((1, 2, 0))
        gt_depth = gt_depth * obj_mask  # non-object area have zero depth
        c2w = keyframe["c2w"]

        pts = create_point_cloud(gt_color, 1.005 * gt_depth, intrinsics, c2w)
        flat_gt_depth = gt_depth.flatten()
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in gt_depth
        pts = pts[non_zero_depth_mask]

        if is_new:
            if self.uniform_seed_interval < 0:
                uniform_ids = np.arange(pts.shape[0])
            else:
                num = np.int32(pts.shape[0] / self.uniform_seed_interval)
                uniform_ids = np.random.choice(pts.shape[0], num, replace=False)    # sample points uniformly
            sample_ids = uniform_ids
        else:
            # @TODO: implement non-new map case
            sample_ids = np.arange(0)

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
