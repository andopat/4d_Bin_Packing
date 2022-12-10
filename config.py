#####
##### Prediction input/output file names #####
N_PARALLEL_JOBS = 16 # set num cores for multi-processing..more the better
ITEM_MASTER_FILE = "./Data/ItemMaster.xlsx" # needed for Item Dimensions info for customer order
##### Prediction input/output file names #####
#####


#####
##### Some thresholds for faster results
min_supported_corners = 3 # eg how many corners need to be directly supported from below.. 2 less stable but may be more tight
rl_threshold_num_containers = 6 # when to use RL over heuristic
#####



#####
##### RL model paths #####
save_dir  = './train/save_dir/' 
epoch_dir = './train/epochs_dir/'
save_model_name = "model.pt"
##### RL model paths #####
#####


#####
##### Packing configuration related #####
from containers_info  import UNIT_LWH, CONTAINERS_CONFIG

lwh_unit = UNIT_LWH
allow_rotations = [0, 1, 2, 3] # 0: no rotation, 1:X<->Y, 2:Y<->Z, 3:X<->Z (see envs/box.py for more)

num_rotations = len(allow_rotations)
num_containers = CONTAINERS_CONFIG["num_containers"] - 3 # for RL model ignore the ENV boxes as their one side is < 1 inch
max_X = CONTAINERS_CONFIG["max_X"]
max_Y = CONTAINERS_CONFIG["max_Y"]
max_Z = CONTAINERS_CONFIG["max_Z"]
max_W = CONTAINERS_CONFIG["max_W"]

x_placement_poss = list(range(0, int(max_X))) # reduce action space dimensionality by reducing placement x/y grud points (8)
y_placement_poss = list(range(0, int(max_Y))) 


num_items = 10 
lookahead = 0 # how many more box's info available along with current box info when packing current box
dummy_container= "NA"
##### Packing configuration related #####
#####



###################################################################
########### Model's action space & observation space related ######
# XY grid stacked & flattened..reshaped later during training with CNN
container_hwv_maps_len = num_containers*3*max_X*max_Y # for each container XY grid of height_map, free_weight_map, free_volume_map
current_looakhead_items_len = 5*max_X*max_Y 
act_len = num_containers*num_rotations*len(x_placement_poss)*len(y_placement_poss)# taking frontest y as available from feasible positions
obs_mask_len = num_containers*num_rotations*max_X*max_Y
obs_len = container_hwv_maps_len + current_looakhead_items_len + obs_mask_len
channel = num_containers*3 +  5 + num_containers*num_rotations + 2
pred_mask_len = act_len

actor_hidden = 100
critic_hidden = 100
use_container_ids = [-1]
###################################################################



####################################################################
############# Training & Model related ############################
num_processes = 2 #int(16) # how many training CPU processes to use for training - based on CPU RAM

num_iterations    = 1000
num_trainEpisodes = 15
num_testEpisodes  = 30

search_depth     = max(1, num_items)
simulation_times = 3
gamma = 1 # discount factor for rewards (default: 1)

batch_size = 32
epochs = 50


load_model = False


