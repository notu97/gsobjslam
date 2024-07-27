import ultralytics.engine.results
from ultralytics import YOLO

from utils.utils import *
from datasets import *
from associator import Associator
from mapper import Mapper


class GS_OBJ_SLAM(object):

    def __init__(self, configs_path: str) -> None:

        self.configs = load_config(configs_path)
        self.dataset = get_dataset(self.configs['dataset_name'])(self.configs)
        self.yolo = YOLO("yolo_models/yolov8n-seg.pt")  # load an official model
        self.associator = Associator(self.configs)
        self.mapper = Mapper(self.configs)

        self.gaussian_models = []

    def track(self, rgb) -> ultralytics.engine.results.Results:

        tensor_image = torch.tensor(rgb, dtype=torch.float32)
        tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0)   # shape (1, 3, H, W)
        results = self.yolo.track(tensor_image, persist=True)

        return results[0]

    def run(self) -> None:

        for frame_id in range(len(self.dataset)):
            if self.configs['gt_camera']:
                estimated_c2w = self.dataset[frame_id][-1]
            else:
                raise NotImplementedError

            yolo_result = self.track(self.dataset[frame_id][1])
            association = self.associator.associate(yolo_result, self.gaussian_models)
            # iterate over objects
            for i in range(len(association)):
                associated_map_idx = association[i]
                if 0 <= associated_map_idx < len(self.gaussian_models):
                    self.mapper.optimize_map(estimated_c2w, yolo_result, self.gaussian_models[associated_map_idx], i)
                else:
                    self.gaussian_models.append(self.mapper.new_map(estimated_c2w, yolo_result, i))
