import ultralytics.engine.results
from argparse import ArgumentParser
import torchvision

from gaussian_model import GaussianModel
from arguments import OptimizationParams
from datasets import *
from utils.utils import *


class Mapper:

    def __init__(self, config: dict, dataset: BaseDataset) -> None:

        self.config = config
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
        color_transform = torchvision.transforms.ToTensor()
        keyframe = {
            "color": color_transform(gt_color).cuda(),
            "depth": np2torch(gt_depth, device="cuda"),
            "render_settings": get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, w2c)}

        return submap
