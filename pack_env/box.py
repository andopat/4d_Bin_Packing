"""
Box configurations (terms box & item are used interchangeably eg they represent a single item from customer order to be packed)

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
import copy

class Box(object):
    def __init__(self, x=None, y=None, z=None, dx=None, dy=None, dz=None, wt=None, name=None, parent_gen=None, orig_size=None, orig_intXY_sort_size=None, pack_cntr_id=None, pack_cntr_name=None, pack_cntr_size=None, pack_rot=0):        
        """
        dx,dy,dz : size of the item/box in inches
        x,y,z    : the position of the front-left-bottom (FLB) corner of the box
        wt       : weight of the item/box
        """
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.wt = wt
        self.parent_gen = parent_gen # for tracking which container was used to generate the box during artifical training/testing data generation or customer order_id in prediction
        self.name = name # to track item name during prediction
        self.orig_size = orig_size # during prediction, we integrize X & Y values, keep prior original ones too for sake of tracking
        self.orig_intXY_sort_size = orig_intXY_sort_size # after rotation dimensions
        self.pack_rot = pack_rot # if box was rotated before packing
        self.pack_cntr_id = pack_cntr_id # to track which container was it packed in
        self.pack_cntr_name = pack_cntr_name     
        self.pack_cntr_size = pack_cntr_size  

        self.rotate_NOOP = 0 # no rotation
        self.rotate_XY = 1 # X<->Y, 
        self.rotate_XZ = 2 # X<->Z, 
        self.rotate_YZ = 3 # Y<->Z

        self.rotation_lookUp = {0:"None", 1:"X<->Y", 2:"X<->Z", 3:"Y<->Z"}

    def vol(self):
        return self.dx*self.dy*self.dz

    def standardize(self):
        return tuple([self.x, self.y, self.z, self.dx, self.dy, self.dz, self.wt])
    
    def rotate(self, rotation):
        """
        Rotate this Box in place
        """
        if (rotation == self.rotate_XY): self.dx, self.dy = self.dy, self.dx
        if (rotation == self.rotate_XZ): self.dx, self.dz = self.dz, self.dx
        if (rotation == self.rotate_YZ): self.dy, self.dz = self.dz, self.dy
    
    def __repr__(self):
        """
        String representation
        """
        desc = "|***Box: Parent({}); Name({}); OrigSize(LWH):{}; Sorted_IntXY_Size:{}; Weight({}); PackInfo => Container-id:{}/name:{}/size:{}; Rotation:{}/{}, Position:{}; PackSize(LWH):{}***|"\
                .format(self.parent_gen, self.name, self.orig_size, self.orig_intXY_sort_size, self.wt, self.pack_cntr_id, self.pack_cntr_name, self.pack_cntr_size, self.pack_rot, self.rotation_lookUp[self.pack_rot], (self.x, self.y, self.z), (self.dx, self.dy, self.dz))
        return desc

    def basic_info(self):
        desc = self.name + " Pos:{}".format((np.round(self.x,3), np.round(self.y,3), np.round(self.z,3))) + " Pack_XYZ:{}".format((self.dx, self.dy, self.dz))
        return desc