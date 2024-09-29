import ultralytics.engine.results
from ultralytics import YOLO
import torchvision
from itertools import groupby
from spatialmath import SE3, UnitQuaternion
from spatialmath.base import trnorm


from utils.utils import *
from datasets import *
from associator import Associator
from mapper import Mapper
from logger import Logger

# Quadricslam imports
from typing import Callable, Dict, List, Optional, Union
import gtsam
import gtsam_quadrics
from quadricslam_states import QuadricSlamState, StepState, SystemState, Detection, qi as QI
from quadric_utils import (
    QuadricInitialiser,
    initialise_quadric_from_depth,
    new_factors,
    new_values,
)
import numpy as np
from spatialmath import SE3
import pickle

class GS_OBJ_SLAM(object):

    def __init__(self, 
                 configs_path: str,
                 initial_pose: Optional[SE3] = None,
                 noise_prior: np.ndarray = np.array([0] * 6, dtype=np.float64),
                 noise_odom: np.ndarray = np.array([0.01] * 6, dtype=np.float64),
                 noise_boxes: np.ndarray = np.array([3] * 4, dtype=np.float64),
                 optimiser_batch: Optional[bool] = None,
                 optimiser_params: Optional[Union[gtsam.ISAM2Params,
                                                gtsam.LevenbergMarquardtParams,
                                                gtsam.GaussNewtonParams]] = None,
                 on_new_estimate: Optional[Callable[[QuadricSlamState], None]] = None,
                 quadric_initialiser:
                 QuadricInitialiser = initialise_quadric_from_depth
        ) -> None:

        self.configs = load_config(configs_path)
        self.dataset = get_dataset(self.configs['dataset_name'])(self.configs)
        self.yolo = YOLO("yolo_models/yolov8x-seg.pt")  # load an official model
        self.associator = Associator(self.configs)
        self.mapper = Mapper(self.configs, self.dataset)
        self.logger = Logger(self.configs['output_path'])

        self.gaussian_models = []   # each object as a submap
        self.associations = {}  # key: tracking id from YOLO. value: idx of submap.

        self.cur_gt_color = None
        self.cur_gt_depth = None
        self.cur_est_c2w = None

        self.on_new_estimate = on_new_estimate
        self.quadric_initialiser = quadric_initialiser

        # Bail if optimiser settings and modes aren't compatible
        if (optimiser_batch == True and
                type(optimiser_params) == gtsam.ISAM2Params):
            raise ValueError("ERROR: Can't run batch mode with '%s' params." %
                             type(optimiser_params))
        elif (optimiser_batch == False and optimiser_params is not None and
              type(optimiser_params) != gtsam.ISAM2Params):
            raise ValueError(
                "ERROR: Can't run incremental mode with '%s' params." %
                type(optimiser_params))
        if optimiser_params is None:
            optimiser_params = (gtsam.LevenbergMarquardtParams()
                                if optimiser_batch is True else
                                gtsam.ISAM2Params())

        # Setup the system state, and perform a reset
        self.state = QuadricSlamState(
            SystemState(
                initial_pose=SE3() if initial_pose is None else initial_pose,
                noise_prior=noise_prior,
                noise_odom=noise_odom,
                noise_boxes=noise_boxes,
                optimiser_batch=type(optimiser_params) != gtsam.ISAM2Params,
                optimiser_params=optimiser_params))
        self.reset()
    
    def reset(self) -> None:
        # self.data_source.restart()

        s = self.state.system
        s.associated = []
        s.unassociated = []
        s.labels = {}
        s.graph = gtsam.NonlinearFactorGraph()
        s.estimates = gtsam.Values()
        s.optimiser = (None if s.optimiser_batch else s.optimiser_type(
            s.optimiser_params))

        s.calib_depth = 1
        s.calib_rgb = np.array([520.9, 521.0, 0, 325.1, 249.7]) # currently only for freiburg2

        self.state.prev_step = None
        self.state.this_step = None

    def track_objects(self, rgb) -> ultralytics.engine.results.Results:

        results = self.yolo.track(rgb, persist=True)

        return results[0]

    def update_associations(self, yolo_result: ultralytics.engine.results.Results) -> dict:

        ids = list(np.int32(yolo_result.boxes.id.numpy()))
        new_detections = yolo_result[[i for i, id in enumerate(ids) if id not in self.associations.keys()]]
        old_ids = [id for id in ids if id in self.associations.keys()] # Old submap Ids
        associated_models_idxs = [self.associations[id] for id in old_ids] # 
        dangling_models = [(i, model) for i, model in enumerate(self.gaussian_models)
                           if i not in associated_models_idxs]
        
        print("Yolov ids: ", ids)
        print("new_detections: ", new_detections)
        print("old_ids: ",old_ids)
        print("associated_models_idxs: ",associated_models_idxs)
        print("dangling_models: ", dangling_models)

        if (len(dangling_models) == 0) or (len(new_detections) == 0):
            return {}

        new_associations = self.associator.associate(new_detections, dangling_models, self.cur_gt_color,
                                                     self.cur_gt_depth, self.cur_est_c2w, self.dataset.intrinsics) # 3D iou Threshold check is done here
        self.associations.update(new_associations)
        # print("New associations found:")
        # print(new_associations)

        return new_associations
    
    def add_yolo_result_2_quadraicSLAM_state(self, yolo_result):
        # TODO: Add detections to QuadricSLAM states
        pass

    def guess_initial_values(self) -> None:
        # Guessing approach (only guess values that don't already have an
        # estimate):
        # - guess poses using dead reckoning
        # - guess quadrics using Euclidean mean of all observations
        s = self.state.system

        fs = [s.graph.at(i) for i in range(0, s.graph.nrFactors())]

        # Start with prior factors
        for pf in [
                f for f in fs if type(f) == gtsam.PriorFactorPose3 and
                not s.estimates.exists(f.keys()[0])
        ]:
            s.estimates.insert(pf.keys()[0], pf.prior())

        # Add all between factors one-by-one (should never be any remaining,
        # but if they are just dump them at the origin after the main loop)
        bfs = [f for f in fs if type(f) == gtsam.BetweenFactorPose3]
        done = False
        while not done:
            bf = next((f for f in bfs if s.estimates.exists(f.keys()[0]) and
                       not s.estimates.exists(f.keys()[1])), None)
            if bf is None:
                done = True
                continue
            s.estimates.insert(
                bf.keys()[1],
                s.estimates.atPose3(bf.keys()[0]) * bf.measured())
            bfs.remove(bf)
        for bf in [
                f for f in bfs if not all([
                    s.estimates.exists(f.keys()[i])
                    for i in range(0, len(f.keys()))
                ])
        ]:
            s.estimates.insert(bf.keys()[1], gtsam.Pose3())

        # Add all quadric factors
        _ok = lambda x: x.objectKey()
        bbs = sorted([
            f for f in fs if type(f) == gtsam_quadrics.BoundingBoxFactor and
            not s.estimates.exists(f.objectKey())
        ],
                     key=_ok)
        for qbbs in [list(v) for k, v in groupby(bbs, _ok)]:
            self.quadric_initialiser(
                [s.estimates.atPose3(bb.poseKey()) for bb in qbbs],
                [bb.measurement() for bb in qbbs],
                self.state).addToValues(s.estimates, qbbs[0].objectKey())

    def run(self) -> None:
        base = np.array([0,0,0,1])
        gs_mean_history = []
        for frame_id in range(len(self.dataset)):

            # QuadricSLAM state initialization
            # Setup state for the current step
            s = self.state.system
            p = self.state.prev_step
            n = StepState(
                0 if self.state.prev_step is None else self.state.prev_step.i + 1)
            self.state.this_step = n

            _, self.cur_gt_color, self.cur_gt_depth, gt_pose = self.dataset[frame_id]

            # print("gt_pose: ",gt_pose)
            # input("3. Press Enter to continue...")
            # SE3(gt_pose)
            # print("type gt_pose: ",type(gt_pose))

            n.odom = gt_pose
            # print("Odom: ", n.odom)
            # input("2. Press Enter to continue...")
            n.depth = self.cur_gt_depth
            n.rgb = self.cur_gt_color

            if self.configs['gt_camera']:
                self.cur_est_c2w = np.vstack((np.hstack((gt_pose.R,(gt_pose.t).reshape(3,1))),base))
            else:
                raise NotImplementedError

            # track objects
            yolo_result = self.track_objects(self.cur_gt_color)
            if yolo_result.boxes.id is None:
                continue

            # Put Detections from Yolov8 into QuadricSLAM Detection class
            self.add_yolo_result_2_quadraicSLAM_state(yolo_result) # n.detections

            # update associations
            self.update_associations(yolo_result) ## Yolo_trk_id : Submap_id

            # iterate over objects
            for i in range(len(yolo_result)):
                tracking_id = int(yolo_result.boxes.id[i].numpy())
                # if this tracking id is already associated
                if tracking_id in self.associations.keys():
                    # optimize associated submap
                    self.mapper.update(frame_id, self.cur_est_c2w, yolo_result,
                                       self.gaussian_models[self.associations[tracking_id]], i)
                else:
                    '''
                    New Submap Built here, Probably initilize the new quadric here.
                    '''
                    print('New tracking id: %s' % tracking_id)
                    # start a new submap
                    new_gs_submap = self.mapper.new(frame_id, self.cur_est_c2w, yolo_result, i, len(self.gaussian_models))
                    self.gaussian_models.append(new_gs_submap)
                    self.associations[tracking_id] = len(self.gaussian_models) - 1  # record association
                    s.graph.add(
                        gtsam_quadrics.BoundingBoxFactor(
                            gtsam_quadrics.AlignedBox2(new_gs_submap.bounds),
                            gtsam.Cal3_S2(s.calib_rgb), n.pose_key, new_gs_submap.quadric_key,
                            s.noise_boxes))
            
            # print(f"Association: trk_id:submap_id {self.associations}")

            s.labels = {
                d.quadric_key: d.label
                for d in self.gaussian_models
                if d.quadric_key is not None
            }

            # print("Labels: ", s.labels)
            # input("1. Press Enter to continue...")


            # # Add new pose to the factor graph
            if p is None:
                s.graph.add(
                    gtsam.PriorFactorPose3(n.pose_key, s.initial_pose,
                                        s.noise_prior))
            else:
                s.graph.add(
                    gtsam.BetweenFactorPose3(
                        p.pose_key, n.pose_key,
                        gtsam.Pose3(((SE3() if p.odom is None else p.odom).inv() * (SE3() if n.odom is None else n.odom)).A),
                        s.noise_odom))
            
            # Optimise if we're in iterative mode
            if not s.optimiser_batch:
                self.guess_initial_values()
                if s.optimiser is None:
                    s.optimiser = s.optimiser_type(s.optimiser_params)
                # print("HERE")
                s.graph.saveGraph("/home/shiladitya/Projects/gsobjslam/graph_new.dot")
                input("press a key: 1")
                try:
                    # pu.db
                    s.optimiser.update(
                        new_factors(s.graph, s.optimiser.getFactorsUnsafe()),
                        new_values(s.estimates,
                                s.optimiser.getLinearizationPoint()))
                    s.estimates = s.optimiser.calculateEstimate()
                except RuntimeError as e:
                    # For handling gtsam::InderminantLinearSystemException:
                    #   https://gtsam.org/doxygen/a03816.html
                    pass
                s.graph.saveGraph("/home/shiladitya/Projects/gsobjslam/graph_adding_factors.dot")
                input("press a key: 2")
                if self.on_new_estimate:
                    self.on_new_estimate(self.state)

            self.state.prev_step = n

            temp = []
            for gs in self.gaussian_models:
                # print( gs.get_xyz().mean(dim=0))
                temp.append(gs.get_xyz().mean(dim=0))
            gs_mean_history.append(temp)

            # Visualise the mapping for the current frame
            w2c = np.linalg.inv(self.cur_est_c2w)
            color_transform = torchvision.transforms.ToTensor()
            keyframe = {
                "color": color_transform(self.cur_gt_color).cuda(),
                "depth": np2torch(self.cur_gt_depth, device="cuda"),
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
                    keyframe["depth"].unsqueeze(-1),
                    yolo_result)
                
                   
        if self.state.system.optimiser_batch:
            self.guess_initial_values()
            s = self.state.system
            # s.graph.saveGraph("/home/shiladitya/Projects/gtsam_exps/graph_new.dot")
            s.optimiser = s.optimiser_type(s.graph, s.estimates,
                                           s.optimiser_params)
            s.estimates = s.optimiser.optimize()
            if self.on_new_estimate:
                self.on_new_estimate(self.state)
                
                input("press a key: 3")
            
