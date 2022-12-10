"""
Container configuration/manipulation module: 
Box/Item (s) are placed in these containers. Details of full set of containers used for packaging are given in containers_info.py

To make each container's XY grid of same dims we intermediately use a dummy bigger container which contains the container fully & its extra space is marked unaccessible

LWH/XYZ convention [(0,0,0)=> Front-Left-Bottom corner]:
    x: length       (small x = left               , large x = right)
    y: width/depth  (small y = front (near viewer), large y = deep (away from viewer)
    z: height       (small z = low                , large z = high)

             Z(height)
             |
             |   Y(width)
             |  /
             | /
(FLB:(0,0,0))|/________ X (length)
"""

import numpy as np
import copy, os, sys

from .box import Box

sys.path.append("../")
import config

class Container(object):
    def __init__(self, dx=10, dy=10, dz=10, max_wt=30, max_X=48, max_Y=24, max_Z=20, max_W=50, name=None):
        """
        create a bigger dummy container with LWH (max_X, max_Y, max_Z) for original container with LWH (dx, dy, dz), marked extra space as unaccessible via height_map
        original container:
            dx,dy,dz : size in inches
            max_wt   : maximum weight allowed in the container

        bigger dummy container:
            max_X, max_Y, max_Z: maximum length, width & height from all containers set
            max_W: maximum weight allowed from all containers set
        """    
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.container_name = name
        self.container_max_vol = dx*dy*dz # total volume available in original container in cubic ft        
        self.container_max_wt = max_wt # maximum weight allowed
        self.boxes = [] # list of boxes contained

        self.max_X = max_X
        self.max_Y = max_Y
        self.max_Z = max_Z        
        self.max_W = max_W
        self.big_vol =   max_X*max_Y*max_Z # all volume available in bigger dummy container
        
        self.boxes = [] # list of items/boxes packed in the container
        self.total_box_wts = 0. # total weight of boxes contained in the container
        self.total_box_vols = 0. # total volume of boxes contained in the container
        self.free_wt = self.container_max_wt # keep track of available weight capacity 
        self.free_vol = self.container_max_vol # keep track of available empty space

        # keep track of z-axis accessibility for each of XY grid points eg minimum height at which a new item can be placed for a given (x,y) grid point
        self.height_map = None
        self._init_height_map() # need to make consistent for all containers (eg from container_sets.py with different length & width)
        
    def __repr__(self):
        """
        String representation
        """
        desc = "|^^^Container: Name({}); Max Wt({}); Vol-Real/Max({}:{}); Size:{}; total_box_wts({}); total_box_vols({}); free_wt({}); free_vol({}); \n\tBoxes Contained:{} ^^^|\n"\
                .format(self.container_name, self.container_max_wt, self.container_max_vol, self.big_vol, (self.dx, self.dy, self.dz), self.total_box_wts, self.total_box_vols, self.free_wt, self.free_vol, self.boxes)
        return desc

    def _init_height_map(self):
        # min height available for an item placement
        self.height_map = np.ones((self.max_X, self.max_Y))*(self.max_Z - self.dz) # keep only dz from top empty. makes consistent for all containers
        self.height_map[:, self.dy:] = self.max_Z # invalid y coordinates; unaccessible by marking fully occupied on z-axis
        self.height_map[self.dx:, :] = self.max_Z # invalid x coordinates; unaccessible by marking fully occupied on z-axis


    def reset(self):
        self.boxes = []
        self.total_box_wts = 0.
        self.total_box_vols = 0.
        self.free_wt = self.container_max_wt
        self.free_vol = self.container_max_vol
        self.height_map = None
        self._init_height_map()

    def get_hwv_map(self):
        height_map   = copy.deepcopy(self.height_map)
        free_wt_map  = np.ones((self.max_X, self.max_Y))*self.free_wt
        free_vol_map = np.ones((self.max_X, self.max_Y))*self.free_vol
        return np.stack((height_map, free_wt_map, free_vol_map), axis=0)

    @staticmethod
    def update_height_map(hmap, box):
        le = box.x
        ri = box.x + box.dx
        up = box.y
        do = box.y + box.dy
        max_h = np.max(hmap[le:ri, up:do])
        max_h = max(max_h, box.z + box.dz)
        hmap[le:ri, up:do] = max_h
        return hmap

    def check_box_placement_valid(self, box, pos, checkMode="normal", check_print=False):
        """
        return -1 if placement is invalid
        return height of the box base when placed, if placement is good

        box: obj of type "Box"
        pos: tuple (x,y)
        checkMode: str "normal" or "strict" [at "strict" check the box must be supported 100% below its base]
        """
        
        x, y = pos
        if x+box.dx > self.dx or y+box.dy > self.dy: return -1
        if x < 0 or y < 0: return -1

        rec = self.height_map[x:x+box.dx, y:y+box.dy]
        r00 = rec[ 0, 0]
        r10 = rec[-1, 0]
        r01 = rec[ 0,-1]
        r11 = rec[-1,-1]
        rm = max(r00,r10,r01,r11)
        supportedCorners = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
        if supportedCorners < config.min_supported_corners:
            if check_print:
                print("less than 3 supported corners", box, "=>", (self.container_name, x, y, self.dz, self.max_Z))
            return -1

        max_h = np.max(rec) # box base height if placed here
        assert max_h >= 0
        if max_h + box.dz > self.max_Z:
            if check_print:
                print("max_h violated", box, "\n", (self.container_name, x, y, max_h, self.dz, self.max_Z) )
            return -1

        # check box base is well supported
        max_area = np.sum(rec==max_h)
        area = box.dx * box.dy

        if checkMode == "strict" and max_area<area: return -1

        if max_area/area > 0.95/3: 
            return max_h
        if rm == max_h and supportedCorners == 3 and max_area/area > 0.85/3:
            return max_h
        if rm == max_h and supportedCorners == 4 and max_area/area > 0.50/3:
            return max_h

        if check_print:
            print("max_area violated", box, "\n", (self.container_name, x, y, max_h, self.dz, self.max_Z))
        return -1


    def drop_box(self, box, pos, check_print=False):
        """
        place a box at pos into the container
        """
        # check violation of available space/weight capacity
        if (box.wt > self.free_wt) or (box.vol() > self.free_vol):
            if check_print:
                print("box placement failed because of free weight & volume condition")
            return False, None

        # check if a position is legitimate
        x, y = pos
        new_h = self.check_box_placement_valid(box, (x, y), check_print) # new_h: box base height
        if new_h == -1:
            return False, None

        # place the box, update box's height(z) & update heightmaps
        box.x, box.y, box.z = x, y, new_h
        self.boxes.append(box)

        self.height_map = self.update_height_map(self.height_map, box)

        self.total_box_wts = sum([b.wt for b in self.boxes])
        self.total_box_vols = sum([b.vol() for b in self.boxes])
        self.free_wt = self.container_max_wt - self.total_box_wts
        self.free_vol = self.container_max_vol - self.total_box_vols

        return True, box