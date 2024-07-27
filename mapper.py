import numpy as np
import ultralytics.engine.results

from gaussian_model import GaussianModel


class Mapper:

    def __init__(self, config: dict) -> None:

        self.config = config

    def new_map(self, c2w: np.ndarray, yolo_result: ultralytics.engine.results.Results,
                object_idx: int) -> GaussianModel:

        pass

    def optimize_map(self, c2w: np.ndarray, yolo_result: ultralytics.engine.results.Results,
                     submap: GaussianModel, object_idx: int) -> GaussianModel:

        pass
