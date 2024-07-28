import ultralytics.engine.results
from ultralytics import YOLO
import torchvision

from utils.utils import *
from datasets import *
from associator import Associator
from mapper import Mapper
from logger import Logger


class GS_OBJ_SLAM(object):

    def __init__(self, configs_path: str) -> None:

        self.configs = load_config(configs_path)
        self.dataset = get_dataset(self.configs['dataset_name'])(self.configs)
        self.yolo = YOLO("yolo_models/yolov8n-seg.pt")  # load an official model
        self.associator = Associator(self.configs)
        self.mapper = Mapper(self.configs, self.dataset)
        self.logger = Logger(self.configs['output_path'])

        self.gaussian_models = []   # each object as a submap
        self.associations = {}  # key: tracking id from YOLO. value: idx of submap.

    def track_objects(self, rgb) -> ultralytics.engine.results.Results:

        results = self.yolo.track(rgb, persist=True)

        return results[0]

    def run(self) -> None:

        for frame_id in range(len(self.dataset)):

            _, gt_color, gt_depth, gt_pose = self.dataset[frame_id]
            if self.configs['gt_camera']:
                estimated_c2w = gt_pose
            else:
                raise NotImplementedError

            yolo_result = self.track_objects(gt_color)
            if yolo_result.boxes.id is None:
                continue
            # iterate over objects
            for i in range(len(yolo_result)):
                tracking_id = int(yolo_result.boxes.id[i].numpy())
                # if this tracking id is already seen
                if tracking_id in self.associations.keys():
                    # optimize associated submap
                    self.mapper.update(frame_id, estimated_c2w, yolo_result,
                                       self.gaussian_models[self.associations[tracking_id]], i)
                else:
                    print('New tracking id: %s' % tracking_id)
                    ascn = self.associator.associate(yolo_result, i, self.gaussian_models)  # try to associate
                    if ascn == -1:  # cannot find association
                        # start a new submap
                        self.gaussian_models.append(self.mapper.new(frame_id, estimated_c2w, yolo_result, i))
                        self.associations[tracking_id] = len(self.gaussian_models) - 1  # record association
                    else:
                        self.associations[tracking_id] = ascn   # record association
                        # optimize associated submap
                        self.mapper.update(frame_id, estimated_c2w, yolo_result,
                                           self.gaussian_models[self.associations[tracking_id]], i)

            # Visualise the mapping for the current frame
            w2c = np.linalg.inv(estimated_c2w)
            color_transform = torchvision.transforms.ToTensor()
            keyframe = {
                "color": color_transform(gt_color).cuda(),
                "depth": np2torch(gt_depth, device="cuda"),
                "render_settings": get_render_settings(
                    self.dataset.width, self.dataset.height, self.dataset.intrinsics, w2c)}

            with torch.no_grad():
                render_pkg_vis = render_gs(self.gaussian_models, keyframe['render_settings'])
                image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]

                self.logger.vis_mapping_iteration(
                    frame_id, 0,
                    image_vis.clone().detach().permute(1, 2, 0),
                    depth_vis.clone().detach().permute(1, 2, 0),
                    keyframe["color"].permute(1, 2, 0),
                    keyframe["depth"].unsqueeze(-1))
