from utils.utils import *
from datasets import *


configs_path = 'configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml'
config = load_config(configs_path)

mydata = TUM_RGBD(config)
