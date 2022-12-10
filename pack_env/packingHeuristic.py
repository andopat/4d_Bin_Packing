import numpy as np
import copy
import time, sys

from .box import Box
from .container import Container
from .container_sets import ContainerSets

sys.path.append("../")
import config

class PackHeuristic():
	def __init__(self, items_info, input_box_list, print_t=False):
		'''
		see ../tests/predict.py - pack_customer_order() where items are pre-processed for model input
		items_info contains all item details to be packed
			- item: [l, w, h, dx, dy, dz, box_wt, item_name, orderID, box_lwh]
		input_box_list contains items wrapped in Box
			- item_box: Box(dx=dx, dy=dy, dz=dz, wt=wt, name=item_name, parent_gen=orderID, orig_size=box_lwh, orig_intXY_sort_size=(dx, dy, dz))
		'''
		self.items_info = items_info
		self.input_box_list = input_box_list
		self.print_t = print_t

		self.items_left_to_pack = items_info
		self.input_box_list_left_to_pack = input_box_list
		self.ids_left_to_pack = list(range(len(items_info)))

		self.items_partition_info = []

	@staticmethod
	def update_height_map(hmap, x, y, z, dx, dy, dz):
		max_h = np.max(hmap[x:x+dx, y:y+dy])
		max_h = max(max_h, z + dz)
		hmap[x:x+dx, y:y+dy] = max_h
		return hmap
	
	@staticmethod
	def check_valid_placement(hmap, x, y, z, dx, dy, dz, container_dx, container_dy, container_dz):
		if x + dx > container_dx or y + dy > container_dy: return False
		if x < 0 or y < 0: return False

		rec = hmap[x:x+dx, y:y+dy]
		r00 = rec[ 0, 0]
		r10 = rec[-1, 0]
		r01 = rec[ 0,-1]
		r11 = rec[-1,-1]
		rm = max(r00,r10,r01,r11)
		supportedCorners = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
		if supportedCorners < config.min_supported_corners:
		    return False

		max_h = np.max(rec) # box base height if placed here
		assert max_h >= 0
		if max_h + dz > container_dz:
			return False

		return True

	def check_packing(self):

		some_invalid_item = False
		while len(self.ids_left_to_pack) > 0 and not some_invalid_item:
			# print("items_left_to_pack:{}, ids:{}".format(len(self.ids_left_to_pack), self.ids_left_to_pack))

			keep_recent = [None]
			ids_packed = []
			cant_be_packed = []
			for i in range(1, len(self.ids_left_to_pack)+1):
				ids = [self.ids_left_to_pack[j] for j in range(i) if self.ids_left_to_pack[j] not in cant_be_packed]
				current_items_info = [self.items_info[j] for j in  ids]
				current_input_box_list = [self.input_box_list[j] for j in  ids]

				packed, pack_info = self.check_packing_single_container(current_items_info, current_input_box_list)
				# print("\t i:{} ids:{}, cant_be_packed:{}".format(i, ids, cant_be_packed))

				if packed:
					ids_packed = ids
					keep_recent.append(pack_info)
					# container_name = [k for k in pack_info.keys()][0]
					# print("\t\t packed => ids:{}, len:{}".format(ids, len(pack_info[container_name]["packed_boxes"])))					
				else:
					# print("\t\t\t can't be packed", ids[-1])
					cant_be_packed.append(ids[-1])

			# print("can be packed:{}, cant be packed:{}".format(ids_packed, cant_be_packed))
			if len(cant_be_packed) == len(self.ids_left_to_pack):
				some_invalid_item = True

			self.ids_left_to_pack = cant_be_packed
			self.items_partition_info.append([ids_packed, keep_recent[-1]])

		if len(self.ids_left_to_pack) > 0:
			# in case some LWH or wt violation leading to some item just cant be packed
			return False, [], [], "can't be packed", None

		final_packing_info = {}
		cntrs = {}
		container_wise_packing = ""
		for item in self.items_partition_info:
			ids = item[0]
			item = item[-1]
			container_name = [k for k in item.keys()][0]
			cx = container_name.split("(")[0]
			if cx not in cntrs.keys():
				cntrs.update({cx:0})
			cntrs[cx] += 1
			new_cntr_name = cx + "(" + str(cntrs[cx]) + ")"
			final_packing_info[new_cntr_name] = item[container_name]

			container_wise_packing += new_cntr_name + "<= #items:{} details =>".format(len(item[container_name]["packed_boxes"])) \
									+ str([b.basic_info() for b in item[container_name]["packed_boxes"]]) + "\n"

		container_wise_packing = container_wise_packing[:-1] # remove last newline
		used_container_names = [k for k in final_packing_info.keys()]
		used_container_ids = [val["container_id"] for key, val in final_packing_info.items()]

		return True, used_container_ids, used_container_names, final_packing_info, container_wise_packing



	def check_packing_single_container(self, items_info, input_box_list):

		cntr_details = [[container_id, cntr_info["L"], cntr_info["W"], cntr_info["H"], cntr_info["max_weight"], cntr_info["max_vol"]] \
							for container_id, cntr_info in config.CONTAINERS_CONFIG["container_details"].items()]
		# cntr_details = [[container_id, cntr_info["X"], cntr_info["Y"], cntr_info["H"], cntr_info["max_weight"], cntr_info["max_vol"]] \
		# 					for container_id, cntr_info in config.CONTAINERS_CONFIG["container_details"].items()]

		cntr_details.sort(key=lambda x: x[5])

		max_X = max([item[3] for item in items_info])
		max_Y = max([item[4] for item in items_info])
		max_Z = max([item[2] for item in items_info])		
		combined_vol = sum([item[3]*item[4]*item[2] for item in items_info])	
		combined_W = sum([item[6] for item in items_info])

		### When a single box is enough and be trivially verified fast - find the one with smallest volume ###
		suitable_single_container_ids = [container[0] for container in cntr_details \
											if (combined_vol <= container[5]) and (combined_W <= container[4]) \
												and (max_X <= container[1]) and (max_Y <= container[2]) and (max_Z <= container[3])]

		if len(suitable_single_container_ids) > 0:
			for cid in suitable_single_container_ids:
				placed_container_name = config.CONTAINERS_CONFIG["container_details"][cid]["name"] + "(1)"
				packed, final_packing_info = self.pack_in_single_container(items_info, input_box_list, cid, placed_container_name)
				if packed:
					if len(final_packing_info[placed_container_name]["packed_boxes"]) == len(items_info):
						return True, final_packing_info

		return False, None

	def pack_in_single_container(self, items_info, input_box_list, cid, placed_container_name):

		max_X = max([item[3] for item in items_info])
		max_Y = max([item[4] for item in items_info])
		max_Z = max([item[2] for item in items_info])	
		combined_Z = sum([item[2] for item in items_info])
		combined_W = sum([item[6] for item in items_info])
		combined_vol = sum([item[3]*item[4]*item[2] for item in items_info])	

		container = config.CONTAINERS_CONFIG["container_details"][cid]
		packing_info = {placed_container_name:{"container_id":cid, "packed_boxes":[]}}

		cntr_dx, cntr_dy, cntr_dz = container["X"], container["Y"],  container["H"]
		height_map = np.zeros((cntr_dx, cntr_dy))

		xyz_pos_rot = []
		used_points = []

		flb_corners = [(0,0,0)]
		re_corners  = []
		fu_corners = [] 
		for i in range(len(items_info)):
			l, w, h, dx, dy, dz = items_info[i][:6]

			# avbl_corners = flb_corners + re_corners + fu_corners
			avbl_corners = re_corners + fu_corners + flb_corners
			avbl_corners = [corner for corner in avbl_corners if corner not in used_points]

			# if i <= 5:
			# 	print(flb_corners, ">", re_corners, ">", fu_corners, ">u<", used_points, "=", avbl_corners)

			item_packed = False

			for j in range(len(avbl_corners)):
				x,y,z = avbl_corners[j]

				cndn_1 = dx + x <= container["X"] and dy + y <= container["Y"] and h + z <= container["H"] # without rotation
				cndn_2 = dy + x <= container["X"] and dx + y <= container["Y"] and h + z <= container["H"] # X-Y rotation
				cndn_3 = dz + x <= container["X"] and dy + y <= container["Y"] and dx + z <= container["H"] # X-Z rotation
				cndn_4 = dx + x <= container["X"] and dz + y <= container["Y"] and dy + z <= container["H"] # Y-Z rotation

				check_valid = self.check_valid_placement(height_map, x, y, z, dx, dy, h, cntr_dx, cntr_dy, cntr_dz)

				if check_valid:
					if cndn_1 or cndn_2 or cndn_3 or cndn_4:
						# if i <= 5:
						# 	print("\t", (x,y,z))
						item_packed = True
						used_points.append((x,y,z))
						
					if cndn_1:
						xyz_pos_rot.append([x, y, z, 0])
						flb_corners.append((x, y, h+z))
						re_corners.append((dx+x, y, z))
						fu_corners.append((x, dy+y, z))
						height_map = self.update_height_map(height_map, x, y, z, dx, dy, h)		
						break
					elif cndn_2:
						xyz_pos_rot.append([x, y, z, 1])
						flb_corners.append((x, y, h+z))
						re_corners.append((dy+x, y, z))
						fu_corners.append((x, dx+y, z))
						height_map = self.update_height_map(height_map, y, x, z, dx, dy, h)																								
						break
					elif cndn_3:
						xyz_pos_rot.append([x, y, z, 2])
						flb_corners.append((x, y, l+z))
						re_corners.append((dz+x, y, z))
						fu_corners.append((x, dy+y, z))
						height_map = self.update_height_map(height_map, z, y, x, dz, dy, l)																											
						break
					elif cndn_4:
						xyz_pos_rot.append([x, y, z, 3])
						flb_corners.append((x, y, w+z))
						re_corners.append((dx+x, y, z))
						fu_corners.append((x, dz+y, z))
						height_map = self.update_height_map(height_map, x, z, y, dx, dz, w)											
						break

			if not item_packed:
				break

		if len(xyz_pos_rot) == len(items_info):
			for i in range(len(items_info)):
				l,w,h,dx,dy,dz = items_info[i][:6]

				box =  input_box_list[i]
				box.pack_cntr_id = cid
				box.pack_cntr_name = placed_container_name
				box.pack_cntr_size = (container["L"], container["W"], container["H"])
				box.x = xyz_pos_rot[i][0]
				box.y = xyz_pos_rot[i][1]
				box.z = xyz_pos_rot[i][2]				
				box.pack_rot = xyz_pos_rot[i][3]

				if xyz_pos_rot[i][3] == 1: # X<->Y
					box.dx, box.dy = box.dy, box.dx
				elif xyz_pos_rot[i][3] == 2: # X<->Z
					box.dx, box.dz = box.dz, box.dx
				elif xyz_pos_rot[i][3] == 3: # Y<->Z
					box.dy, box.dz = box.dz, box.dy
													
				packing_info[placed_container_name]["packed_boxes"].append(box)

			return True, packing_info

		return False, None


