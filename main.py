from utils.utils import *
from datasets import *
from GS_OBJ_SLAM import *
from quadricslam.visualisation import visualise


configs_path = 'configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml'
slam = GS_OBJ_SLAM(configs_path, 
                   optimiser_batch= False,
                   on_new_estimate=(lambda state: visualise(state.system.estimates, state.system.
                                    labels, state.system.optimiser_batch)),)
slam.run()

print(slam.associations)
