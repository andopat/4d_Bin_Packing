'''
Creates box list sequences needed by model for packing

CuttingBoxSeqCreator: 
    - for training/testing
    - generate data by recursively cutting boxes from container (s) so that it can be fit there surely

PredictionBoxSeqCreator: 
    - during prediction
    - creates box sequences from provided item list of customer order
'''

import numpy as np
import copy
import random
import sys

from .box import Box
from .container import Container

sys.path.append("../")
import config

class PredictionBoxSeqCreator():
    def __init__(self, customer_order_list=None, seed=None):
        '''
        creates input sequence for model prediction
        customer_order_list => list of items/boxes to be packed in the containers for the resp. customer order
        '''
        # super(PredictionBoxSeqCreator).__init__()

        self.num_items = config.num_items
        self.customer_order_list = copy.deepcopy(customer_order_list)
        self.customer_id = customer_order_list[0].parent_gen
        
        self.allow_rotations  = config.allow_rotations
        self.num_items        = config.num_items
        self.num_rotations    = config.num_rotations
        self.n_forseeable_box = 1 + config.lookahead        
        self.lookahead        = config.lookahead
        
        self.dummy_container = config.dummy_container

        self.box_list = []
        self.original_box_list = []
        self.original_box_list_vol = 0

        self.seed = seed
        if seed is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.reset()

    def reset_list(self, list_, current_id, replace_id):
        current_item = list_.pop(current_id)
        replace_item = list_.pop(replace_id - 1)

        final_list = list_[:current_id] + [replace_item, current_item] + list_[current_id:]

        return final_list

    def reset_box_list(self, current_box_id, replace_box_id, check_print=False):
        # if check_print:
        #     print("\nbefore reset", len(self.box_list))
        self.box_list = self.reset_list(self.box_list, current_box_id, replace_box_id)
        # if check_print:
        #     print("\n after reset\n", len(self.box_list))

    def reset(self):
        self.box_list = []
        self.original_box_list = []
        self.original_box_list_vol = 0
        self.generate_box_list()

    def _rotate_box(self, box):
        rotation = self.rng.choice(self.allow_rotations)
        box.rotate(rotation)
        return box

    def generate_box_list(self):
        self.box_list = self.customer_order_list
        self.num_boxes = len(self.box_list)
        self.full_box_vol = sum([b.vol() for b in self.box_list])


class CuttingBoxSeqCreator():
    """
    Recursively bisect a container (i.e. a big box) to small boxes
    Thus the full box sequence generated is guaranteed to have a perfect way to pack into the container. 
    We can additionally restrict the box sequence to only num_items (eg maximum number of items to be packed for each customer, see config.py)

    allow_rotations : list of Enum Rotate
    n_foreseeable_box : int>=1
    container_zise    : tuple (x,y,z)
    minSideLen : int
    maxSideLen : int
    sortMethod : str, "ByZ" or "ByStackOrder"

    will cut until minSideLen <= box side <= maxSideLen, for all 3 side of the boxes
    """
    def __init__(self):

        self.num_containers = config.num_containers

        self.allow_rotations  = config.allow_rotations
        self.num_items        = config.num_items
        self.num_rotations    = config.num_rotations
        self.n_forseeable_box = 1 + config.lookahead        
        self.lookahead        = config.lookahead


        self.container_id = np.random.choice(config.use_container_ids, 1)[0]
        if self.container_id == -1:
            self.container_id = np.random.choice(range(self.num_containers), size=1)[0]

        self.container_id = 11

        self.container_name = config.CONTAINERS_CONFIG["container_details"][self.container_id]["name"]
        self.container_size = [config.CONTAINERS_CONFIG["container_details"][self.container_id][k] for k in ["X", "Y", "Z"]]
        self.container_max_wt = config.CONTAINERS_CONFIG["container_details"][self.container_id]["max_weight"]
        self.container_vol = (self.container_size[0]*self.container_size[1]*self.container_size[2])/(12*12*12) # cubic ft

        self.dummy_container = config.dummy_container

        self.minSideLen = max(1, int(min(self.container_size)/5) )
        self.maxSideLen = maxSideLen = max(1, int(min(self.container_size)/1.1))
        self.sortMethod = "ByStackOrder" # ["ByZ", "ByStackOrder"]

        assert self.minSideLen*2 <= max(self.container_size)
        assert self.maxSideLen <= min(self.container_size)
        assert self.sortMethod in ["ByZ", "ByStackOrder"]

        self.box_list = []
        self.original_box_list = []
        self.original_box_list_vol = 0

        seed = None
        self.seed = seed
        if seed is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.reset()

    def reset(self):
        self.box_list = []
        self.original_box_list = []
        self.original_box_list_vol = 0
        self.generate_box_list()

    def reset_list(self, list_, current_id, replace_id):
        current_item = list_.pop(current_id)
        replace_item = list_.pop(replace_id - 1)

        final_list = list_[:current_id] + [replace_item, current_item] + list_[current_id:]

        return final_list

    def reset_box_list(self, current_box_id, replace_box_id):
        self.box_list = self.reset_list(self.box_list, current_box_id, replace_box_id)

    def next_box(self):
        return self.box_list[0]

    def next_N_boxes(self):
        return self.box_list[:self.n_foreseeable_box]

    def remaining_boxes(self, from_box_num=0):
        return self.box_list[1:]

    def pop_box(self, idx = 0):
        assert len(self.box_list) > idx
        assert idx < self.n_foreseeable_box
        self.box_list.pop(idx)
    
    def _rotate_box(self, box):
        rotation = self.rng.choice(self.allow_rotations)
        box.rotate(rotation)
        return box

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)

    def generate_box_list(self):

        # we wipe out old boxes, so no mix of boxes from cut last time 
        self.box_list = []
        self.original_box_list = []

        # gen new box seq
        dx, dy, dz = self.container_size[:]
        rawBox = Box(x=0, y=0, z=0, dx=dx, dy=dy, dz=dz, wt=self.container_max_wt, name="cut", parent_gen=self.container_name)        
        self._cut_box(rawBox)        
        self.box_list = [self._rotate_box(b) for b in self.box_list]
        random.shuffle(self.box_list)

        # num_boxes = np.random.choice(range(2, config.num_items+1), 1)[0]    
        # num_boxes = 11
        num_boxes = np.random.choice(range(10, 11), 1)[0]                    
        self.box_list = self.box_list[:num_boxes]
        # self.box_list = self.box_list[:1]
        self.num_boxes = len(self.box_list)    
        # self.box_list[-3] = Box(x=15, y=0, z=0, dx=18, dy=7, dz=4, wt=1.09)
        self.full_box_vol = sum([b.vol() for b in self.box_list])

    def _check_box_size_valid(self, box):
        return  ( (self.minSideLen <= box.dx <= self.maxSideLen) and
                  (self.minSideLen <= box.dy <= self.maxSideLen) and
                  (self.minSideLen <= box.dz <= self.maxSideLen)
                )

    def _cut_box(self, box):
        # enum
        splitX = 0
        splitY = 1
        splitZ = 2
        possibleSplitActions = []
        if box.dx >= self.minSideLen*2 and box.dx > self.maxSideLen: possibleSplitActions.append(splitX)
        if box.dy >= self.minSideLen*2 and box.dy > self.maxSideLen: possibleSplitActions.append(splitY)
        if box.dz >= self.minSideLen*2 and box.dz > self.maxSideLen: possibleSplitActions.append(splitZ)

        if len(possibleSplitActions)==0:
            assert self._check_box_size_valid(box)
            self.box_list.append(box)
            return

        action = self.rng.choice(possibleSplitActions)
        if (action==splitX): pos_range = (self.minSideLen, box.dx - self.minSideLen + 1)
        if (action==splitY): pos_range = (self.minSideLen, box.dy - self.minSideLen + 1)
        if (action==splitZ): pos_range = (self.minSideLen, box.dz - self.minSideLen + 1)
        splitPos = self.rng.integers(pos_range[0], pos_range[1])

        boxA = copy.deepcopy(box)
        boxB = copy.deepcopy(box)

        if (action==splitX):
            boxA.dx, boxB.dx = splitPos, box.dx-splitPos
            boxB.x += splitPos
        elif (action==splitY):
            boxA.dy, boxB.dy = splitPos, box.dy-splitPos
            boxB.y += splitPos
        elif (action==splitZ):
            boxA.dz, boxB.dz = splitPos, box.dz-splitPos
            boxB.z += splitPos
        else:
            print(f"Unknown action {action}")
            raise ValueError

        def get_wt(box_):
            # weight of boxes based on %age volume
            perc_vol = box_.vol()/box.vol()
            box_wt = np.round(box.wt*perc_vol, 2)
            return box_wt
            
        boxA.wt = get_wt(boxA)
        boxB.wt = get_wt(boxB)

        boxA_dims = [boxA.dx, boxA.dy, boxA.dz]
        boxA_dims.sort(reverse=True)
        boxA.dx, boxA.dy, boxA.dz = boxA_dims

        boxB_dims = [boxB.dx, boxB.dy, boxB.dz]
        boxB_dims.sort(reverse=True)
        boxB.dx, boxB.dy, boxB.dz = boxB_dims

        self._cut_box(boxA)
        self._cut_box(boxB)