import ultralytics.engine.results
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from gaussian_model import GaussianModel
from utils.mapper_utils import create_point_cloud


def bboxes_from_tracker(yolo_result: ultralytics.engine.results.Results, gt_color: np.ndarray,
                      gt_depth: np.ndarray, c2w: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:

    bboxes = []
    for i in range(len(yolo_result)):
        obj_mask = yolo_result.masks.data[i, :, :].cpu().detach().numpy()
        gt_color = (gt_color.transpose((2, 0, 1)) * obj_mask).transpose((1, 2, 0))
        gt_depth = gt_depth * obj_mask  # non-object area have zero depth
        pts = create_point_cloud(gt_color, 1.005 * gt_depth, intrinsics, c2w)

        flat_gt_depth = gt_depth.flatten()
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in gt_depth
        pts = pts[non_zero_depth_mask][:, :3]
        
        if not np.any(pts):
            continue

        # bbox: shape ( , 6), representing (xmin, ymin, zmin, xmax, ymax, zmax)
        bbox = np.hstack((np.min(pts, axis=0), np.max(pts, axis=0)))
        bboxes.append(bbox)

    return np.array(bboxes)     # shape (N, 6)


def bboxes_from_gaussians(gaussian_models: list) -> np.ndarray:

    bboxes = []
    for model in gaussian_models:
        pts = model.get_xyz().detach().cpu().numpy()
        # bbox: shape ( , 6), representing (xmin, ymin, zmin, xmax, ymax, zmax)
        bbox = np.hstack((np.min(pts, axis=0), np.max(pts, axis=0)))
        bboxes.append(bbox)

    return np.array(bboxes)     # shape (N, 6)


def iou_3d_batch(bboxes1, bboxes2) -> np.ndarray:

    bboxes2 = np.expand_dims(bboxes2, 0)    # N1 bboxes
    bboxes1 = np.expand_dims(bboxes1, 1)    # N2 bboxes

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    zz1 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    xx2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    yy2 = np.minimum(bboxes1[..., 4], bboxes2[..., 4])
    zz2 = np.minimum(bboxes1[..., 5], bboxes2[..., 5])
    a = np.maximum(0., xx2 - xx1)
    b = np.maximum(0., yy2 - yy1)
    c = np.maximum(0., zz2 - zz1)
    vi = a * b * c
    v1 = ((bboxes1[..., 3] - bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1])
          * (bboxes1[..., 5] - bboxes1[..., 2]))
    v2 = ((bboxes2[..., 3] - bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1])
          * (bboxes2[..., 5] - bboxes2[..., 2]))
    o = vi / (v1 + v2 - vi)

    return o    # shape (N1, N2)


def linear_assignment(cost_matrix) -> np.ndarray:

    x, y = linear_sum_assignment(cost_matrix)

    return np.stack([x, y], axis=1)
