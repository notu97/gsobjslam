import numpy as np
import ultralytics.engine.results
from argparse import ArgumentParser
import torchvision
import time

from gaussian_model import GaussianModel
from arguments import OptimizationParams
from datasets import *
from utils.utils import *
from utils.mapper_utils import *
from losses import *
from quadricslam_states import Detection, QuadricSlamState, qi as QI


class Mapper:

    def __init__(self, config: dict, dataset: BaseDataset) -> None:

        self.config = config
        self.alpha_thre = config['alpha_thre']
        self.uniform_seed_interval = config['uniform_seed_interval']
        self.iterations = config['iterations']
        self.new_submap_iterations = config['new_submap_iterations']
        self.pruning_thre = config['pruning_thre']

        self.dataset = dataset
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

    def new(self, frame_id: int, c2w: np.ndarray, yolo_result: ultralytics.engine.results.Results,
            object_idx: int, submap_id: int) -> GaussianModel:

        gs = GaussianModel(0)
        gs.training_setup(self.opt)
        gs = self.update(frame_id, c2w, yolo_result, gs, object_idx, is_new=True)
        gs.quadric_key = QI(submap_id)

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
        pts = self.seed_new_gaussians(keyframe, seeding_mask, self.dataset.intrinsics, is_new)
        new_pts_num = self.grow_submap(c2w, submap, pts)
        # print("New points num: %d" % new_pts_num)

        max_iterations = self.iterations
        if is_new:
            max_iterations = self.new_submap_iterations
        opt_dict = self.optimize_submap([(frame_id, keyframe)], submap, max_iterations)
        optimization_time = opt_dict['optimization_time']
        print("Optimization time: ", optimization_time)

        # set the bounding box coords
        submap.bounds = yolo_result.boxes[object_idx].xyxy.squeeze()

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

    def seed_new_gaussians(self, keyframe: dict, seeding_mask: np.ndarray,
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
                try:
                    sample_ids = np.random.choice(valid_ids, num, replace=False)
                except ValueError:
                    sample_ids = valid_ids
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

    def optimize_submap(self, keyframes: list, submap: GaussianModel, iterations: int = 100) -> dict:

        iteration = 0
        losses_dict = {}

        start_time = time.time()
        while iteration < iterations + 1:
            submap.optimizer.zero_grad(set_to_none=True)
            # @TODO: optimize using multiple views
            keyframe_id = 0

            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gs([submap], keyframe["render_settings"])

            image, depth = render_pkg["color"], render_pkg["depth"]
            color_transform = torchvision.transforms.ToTensor()
            gt_image = color_transform(keyframe["color"]).cuda() / 255.0
            gt_depth = np2torch(keyframe["depth"], device='cuda')

            valid_mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            obj_mask = torch.from_numpy(keyframe["mask"]).bool().cuda()
            mask = ~obj_mask | valid_mask
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                image[:, mask], gt_image[:, mask]) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            depth_loss = l1_loss(depth[:, mask], gt_depth[mask])

            reg_loss = isotropic_loss(submap.get_scaling())
            total_loss = color_loss + depth_loss + reg_loss
            total_loss.backward()

            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                     "total_loss": total_loss.item()}

            with torch.no_grad():

                if iteration == iterations // 2 or iteration == iterations:
                    prune_mask = (submap.get_opacity()
                                  < self.pruning_thre).squeeze()
                    submap.prune_points(prune_mask)

                # Optimizer step
                if iteration < iterations:
                    submap.optimizer.step()
                submap.optimizer.zero_grad(set_to_none=True)

            iteration += 1
        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict
