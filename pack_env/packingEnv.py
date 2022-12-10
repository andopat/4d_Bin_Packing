import numpy as np
import copy
import time, sys

from .box import Box
from .container import Container
from .container_sets import ContainerSets
from .box_seq_generator import CuttingBoxSeqCreator, PredictionBoxSeqCreator

sys.path.append("../")
import config

class PackEnv():
    def __init__(self, datagen_mode="train", customer_order_list=None, init_container_ids_list=None, **kwargs):

        self.datagen_mode        = datagen_mode
        self.customer_order_list = customer_order_list
        self.init_container_ids_list = init_container_ids_list
        self.sort_init_container_ids()

        self.num_containers = config.num_containers
        self.x_poss = config.x_placement_poss
        self.y_poss = config.y_placement_poss
        self.num_X = len(self.x_poss)
        self.num_Y = len(self.y_poss)               
        self.max_X = config.max_X
        self.max_Y = config.max_Y
        self.max_Z = config.max_Z
        self.max_W = config.max_W

        self.allow_rotations  = config.allow_rotations
        self.num_items        = config.num_items
        self.num_rotations    = config.num_rotations
        self.n_forseeable_box = 1 + config.lookahead

        # Action space dimensionality for each item: (#containers)(#rotations)(#x-discretization steps)(#y-discretization steps)
        self.actionId_lookUp = {}
        actionId = 0
        for i in range(self.num_containers):
            for j in range(self.num_rotations):
                for x in self.x_poss: 
                    for y in self.y_poss:
                        self.actionId_lookUp[actionId] = (i, j, x, y) # converts 0....(config.act_len-1) to (container, rotation, x, y)
                        actionId += 1

        # set up packing related states
        if self.datagen_mode == "train":
            self.boxSeqGenerator = CuttingBoxSeqCreator()                               
        elif self.datagen_mode == "test":
            self.boxSeqGenerator = TestingBoxSeqCreator()                                           
        else:
            self.boxSeqGenerator = PredictionBoxSeqCreator(self.customer_order_list)                               

        self.container_sets_status = ContainerSets(self.boxSeqGenerator.box_list)

        self.current_box_id = 0      
        self.packed_box_counter = 0

        self.used_containers = []

        self.current_container = self.init_container_ids_list[0]


    def reset(self, check_print=False, mode_mcts_sim=True):

        if self.datagen_mode == "train":
            self.boxSeqGenerator = CuttingBoxSeqCreator()                               
        elif self.datagen_mode == "test":
            self.boxSeqGenerator = TestingBoxSeqCreator()                                           
        else:
            self.boxSeqGenerator = PredictionBoxSeqCreator(self.customer_order_list)                               

        self.container_sets_status = ContainerSets(self.boxSeqGenerator.box_list)

        self.current_box_id = 0      
        self.packed_box_counter = 0 # how many got successfully packed

        self.used_containers = []

        self.sort_init_container_ids()
        self.current_container = self.init_container_ids_list[0]

        self.set_cur_observation_vals(check_print, mode_mcts_sim)

    def sort_init_container_ids(self):
        '''
        biggest to smallest container
        '''
        v = []
        for container_id in self.init_container_ids_list:
            vol = config.CONTAINERS_CONFIG["container_details"][container_id]["max_vol"]
            v.append([container_id, vol])
        v.sort(key = lambda x: x[1])
        v = v[::-1]
        self.init_container_ids_list = [v_[0] for v_ in v]


    def step(self, action, check_print=False, mode_mcts_sim=True):
        action = self.actionId_lookUp[action]
        container_id, rotation, x, y = action[0], action[1], action[2], action[3]
        selected_actions = (container_id, rotation, x, y)

        # current box to be placed
        box = self.current_box
        if (rotation == box.rotate_XY): box.dx, box.dy = box.dy, box.dx # 1: X<->Y
        if (rotation == box.rotate_XZ): box.dx, box.dz = box.dz, box.dx # 2: X<->Z
        if (rotation == box.rotate_YZ): box.dy, box.dz = box.dz, box.dy # 3: Y<->Z

        succeeded, box_packed = self.container_sets_status.drop_box(self.current_box_id, container_id, box, (x,y), selected_actions, self.used_containers, check_print, mode_mcts_sim)

        if succeeded:
            self.current_packed_box = box_packed
            self.current_box_id += 1
            self.packed_box_counter += 1

            self.used_containers.append(container_id)
            self.used_containers = list(set(self.used_containers))

            reward = 0
            done = False

            if self.packed_box_counter == self.boxSeqGenerator.num_boxes: # all boxes packed
                reward = 1/len(self.container_sets_status.container_placedBox_lookUp.keys()) 
                # reward = - len(self.container_sets_status.container_placedBox_lookUp.keys()) # number of containers used
                done = True

        else:
            # print("failed", self.packed_box_counter, "\ncont:", self.container_sets_status.containers[container_id], "\nbox:", box, "\n", selected_actions)
            reward = -1 
            done = True


        num_unpacked_boxes = self.boxSeqGenerator.num_boxes  - self.packed_box_counter   
        num_containers = len(self.container_sets_status.container_placedBox_lookUp.keys()) # number of containers used till so far

        frac_packed_boxes = self.packed_box_counter / self.boxSeqGenerator.num_boxes
        info = {'num_total_boxes':self.boxSeqGenerator.num_boxes, 'num_packed':self.packed_box_counter,\
                'num_unpacked':num_unpacked_boxes, 'frac_packed':frac_packed_boxes, "num_containers":num_containers}

        if done:
            return reward, True, info            
        else:
            self.set_cur_observation_vals(check_print)
            return reward, False, info
         

    def set_cur_observation_vals(self, check_print=False, mode_mcts_sim=True):
        # check_print = True
        # if check_print:
        #     print("CURRENT_BOX_ID:{}, len list:{}".format(self.current_box_id, len(self.boxSeqGenerator.box_list)))

        start = time.time()
        self.current_box_mask  = self.get_current_box_mask(self.current_box_id)
        # print("\t\t\ttime for mask calculation", time.time() - start)
        # input("=checking mask time")

        if np.sum(self.current_box_mask) == 0:
            if check_print:
                print("\n**checking whether shuffling items needed or replacing containers", self.current_box_id)

            replace_containers_flag = True
            for box_id in range(self.current_box_id+1, len(self.boxSeqGenerator.box_list)):
                current_box_mask = self.get_current_box_mask(box_id)
                # print("\tisnide", box_id, np.sum(current_box_mask))

                if np.sum(current_box_mask) > 0:
                    if check_print:
                        cb_name = self.boxSeqGenerator.box_list[self.current_box_id].name
                        rb_name = self.boxSeqGenerator.box_list[box_id].name
                        print("\t\tshuffling:  ids:{}<=>{}, names:{}<=>{}".format(self.current_box_id, box_id, cb_name, rb_name))

                    print("b4", [b.name for b in self.boxSeqGenerator.box_list])
                    self.boxSeqGenerator.reset_box_list(self.current_box_id, box_id, check_print)
                    print("after", [b.name for b in self.boxSeqGenerator.box_list])

                    replace_containers_flag = False
                    break

            if replace_containers_flag:
                # start = time.time()
                if check_print:
                    container_names = [key for key, _ in self.container_sets_status.container_placedBox_lookUp.items()]
                    print("\t\t before replacing:", container_names)

                suitable_container_id = self.init_container_ids_list[0]
                self.container_sets_status.replace_containers(suitable_container_id, check_print)

                if check_print:
                    container_names = [key for key, _ in self.container_sets_status.container_placedBox_lookUp.items()]                    
                    print("\t\tafter replacing:", suitable_container_id, "=>", container_names)

            self.current_box_mask  = self.get_current_box_mask(self.current_box_id)
            assert np.sum(self.current_box_mask) > 0.0, "couldn't find any suitable new container, check current box dims"

        self.current_box  = self.boxSeqGenerator.box_list[self.current_box_id]
        remainingBoxes    = self.boxSeqGenerator.box_list[self.current_box_id+1:]


        # ##### model input for CNN
        combined_hwv_map      = self.container_sets_status.get_all_containers_hwv_map() # current state of packing of all containers
        current_box_hwv_map   = self.get_box_hwv_map(self.current_box)
        remainingBoxes_wv_map = self.remainingBoxes_wv_map(remainingBoxes)

        current_box_mask = self.current_box_mask.reshape(-1, self.max_X, self.max_Y)
        self.current_obs = np.concatenate([combined_hwv_map, current_box_hwv_map, remainingBoxes_wv_map, current_box_mask], axis=0)

        self.current_box_mask = current_box_mask.reshape(-1, )

    def get_current_box_mask(self, box_id):
        # num_items = len(self.customer_order_list)
        # if box_id == 0:
        #     allow_rotations = [0]
        # elif 1 <= box_id <= num_items // 3:
        #     allow_rotations = [0, 1]
        # else:
        #     allow_rotations = [0, 1, 2, 3]

        allow_rotations = [0, 1, 2, 3]

        box = self.boxSeqGenerator.box_list[self.current_box_id]

        current_box_mask = self.container_sets_status.get_valid_mask(box, self.init_container_ids_list, allow_rotations)

        # print("=>", current_box_mask.shape, np.sum(current_box_mask), len(current_box_mask))
        return current_box_mask

    def get_box_hwv_map(self, box):
        x_plain   = np.ones((self.max_X, self.max_Y), dtype=np.int32) * box.dx
        y_plain   = np.ones((self.max_X, self.max_Y), dtype=np.int32) * box.dy
        z_plain   = np.ones((self.max_X, self.max_Y), dtype=np.int32) * box.dz
        wt_plain  = np.ones((self.max_X, self.max_Y), dtype=np.float32) * box.wt
        vol_plain = np.ones((self.max_X, self.max_Y), dtype=np.float32) * box.vol()/20 # (box.dx*box.dy*box.dz)

        return np.stack([x_plain, y_plain, z_plain, wt_plain, vol_plain], axis=0)

    def remainingBoxes_wv_map(self, boxes):
        if len(boxes) > 0:
            remaining_weights = sum([b.wt for b in boxes])
            remaining_volumes = sum([b.vol() for b in boxes])
        else:
            remaining_weights = 0.
            remaining_volumes = 0.

        remaining_wmap = np.ones((self.max_X, self.max_Y), dtype=np.float32) * remaining_weights
        remaining_vmap = np.ones((self.max_X, self.max_Y), dtype=np.float32) * remaining_volumes/20
    
        return np.stack([remaining_wmap, remaining_vmap], axis=0)


