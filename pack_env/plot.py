import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import sys
import imageio
from PIL import Image

from pack_env.box import Box

sys.path.append("../")
import config

class Map():
	def __init__(self, customer_order_list, final_packing_list, save_img_file):

		self.customer_order_list = customer_order_list
		self.final_packing_list  = final_packing_list

		self.container_list = []
		self.container_boxes_list = []

		self.save_img_file = save_img_file

		self._init_containers()

		self.animate()

	def _init_containers(self):
		self.container_list, self.container_boxes_list = [], []

		orig_x = 0
		for container_name, val in self.final_packing_list.items():
			cid = val["container_id"]
			container = config.CONTAINERS_CONFIG["container_details"][cid]
			# cbox = Box(x=orig_x, y=0, z= 0, dx=container["X"], dy=container["Y"], dz=container["Z"], wt=container["max_weight"], name=container_name)
			cbox = Box(x=0, y=0, z= 0, dx=container["L"], dy=container["W"], dz=container["H"], wt=container["max_weight"], name=container_name)			
			self.container_list.append(cbox)

			packed_boxes = []
			for b in val["packed_boxes"]:
				b_ = Box(x=b.x, y=b.y, z=b.z, dx=b.dx, dy=b.dy, dz=b.dz, wt=b.wt, orig_size=b.orig_size, orig_intXY_sort_size=b.orig_intXY_sort_size, name=b.name, parent_gen=b.parent_gen, pack_rot=b.pack_rot, pack_cntr_id=b.pack_cntr_id, pack_cntr_name=b.pack_cntr_name, pack_cntr_size=b.pack_cntr_size)
				packed_boxes.append(b_)

			self.container_boxes_list.append(packed_boxes)


	def plot(self, container_list, packed_boxes_list):
		fig = plt.figure(figsize=(8*len(self.container_list), 8))
																		
		axs = []
		for i in range(len(self.container_list)):
			ax_ = fig.add_subplot(1, len(self.container_list), i+1, projection='3d')		
			# box = Box(x=0, y=0, z=0, dx=config.max_X+1, dy=config.max_Y+1, dz=config.max_Z+1)			
			box = Box(x=0, y=0, z=0, dx=config.max_X+1, dy=config.max_X+1, dz=config.max_X+1)			

			self.plot_box(box, ax_, color=(0,0,0,0), showEdges=False) # invisible bound box
			container_name = self.container_list[i].name
			container_wt = self.container_list[i].wt
			container_vol = self.container_list[i].vol()
			container_size = (self.container_list[i].dx, self.container_list[i].dy, self.container_list[i].dz)

			packed_boxes = self.container_boxes_list[i]

			# items_loc = {b.name:"Size:orig{}/sorted_XYZ{} :Pack-in-{}<= Rotate:{}/{}, Pos:{}, Size:{}=>:, Wt:{}".format(b.orig_size, b.orig_intXY_sort_size, b.pack_cntr_name, b.pack_rot, b.rotation_lookUp[b.pack_rot], (b.x, b.y, b.z), (b.dx, b.dy, b.dz), b.wt) for b in packed_boxes}			
			# items_loc = {b.name:"orig_LWH{}:<= Rotate:{}/{}, Pos:{}, PackSize_XYZ:{}=>:, Wt:{}".format(b.orig_size, b.pack_rot, b.rotation_lookUp[b.pack_rot], (np.round(b.x,3), np.round(b.y,3), np.round(b.z,3)), (np.round(b.dx,3), np.round(b.dy,3), np.round(b.dz,3)), np.round(b.wt,3)) for b in packed_boxes}			
			items_loc = {b.name:"orig_LWH{}, Wt:{} <= Pos:{}, Packed_XYZ:{}=>".format(b.orig_size, np.round(b.wt,3), (np.round(b.x,3), np.round(b.y,3), np.round(b.z,3)), (np.round(b.dx,3), np.round(b.dy,3), np.round(b.dz,3))) for b in packed_boxes}			
			
			total_packed_wt = sum([b.wt for b in packed_boxes])
			total_packed_vol = sum([b.vol() for b in packed_boxes])
			title = container_name + "\ncontainer info: XYZ:{}, max wt:{} max vol:{}\n".format(container_size, container_wt, container_vol) + \
							"packing info: total #num of packed-items:{}, total wt:{}, total vol:{}\n".format(len(items_loc), total_packed_wt, np.round(total_packed_vol, 3))
			
			# Add details of a few packed items
			if len(items_loc) > 6:
				title += "Details of only first 6 of total {} packed items dues to space constraint\n".format(len(items_loc))
			else:
				title += "All packed item details\n"

			item_count = 0
			for name, val in items_loc.items():
				title += name + ":" + str(val) + "\n"
				item_count += 1
				if item_count == 6:
					break

			ax_.set_title(title, size=8)
			axs.append(ax_)

		for i in range(len(container_list)):
			self.plot_box(container_list[i], axs[i])
			for j in range(len(packed_boxes_list[i])):
				box = packed_boxes_list[i][j]
				self.plot_box(box, axs[i], color=(0, 0.5, 1, 1))

		fig.canvas.draw()
		image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
		image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		return image_from_plot

	def animate(self):

		with imageio.get_writer(self.save_img_file, fps=1, loop=1) as writer:
			for i in range(1, len(self.container_list)+1):
				fig = self.plot(self.container_list[:i], self.container_boxes_list[:i])
				writer.append_data(fig)


	def plot_parallelepiped(self, cube_definition, ax, color=None, showEdges=True):
		"""
		Draw a 3D parallelepiped to a matplotlib 3d plot
		
		
		cube_definition: corner, plus 3 pts around that corner eg.
				[(0,0,0), (0,1,0), (1,0,0), (0,0,0.1)]
				
		ax: a matplotlib 3d axis obj i.e. from: 
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				
		modified from: https://stackoverflow.com/questions/44881885/python-draw-parallelepiped
		"""
		if color is None: color = (0,0,1,0.1)
			
		cube_definition_array = [np.array(list(item)) for item in cube_definition]

		points = []
		points += cube_definition_array
		vectors = [
			cube_definition_array[1] - cube_definition_array[0],
			cube_definition_array[2] - cube_definition_array[0],
			cube_definition_array[3] - cube_definition_array[0]
		]

		points += [cube_definition_array[0] + vectors[0] + vectors[1]]
		points += [cube_definition_array[0] + vectors[0] + vectors[2]]
		points += [cube_definition_array[0] + vectors[1] + vectors[2]]
		points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

		points = np.array(points)

		edges = [
			[points[0], points[3], points[5], points[1]],
			[points[1], points[5], points[7], points[4]],
			[points[4], points[2], points[6], points[7]],
			[points[2], points[6], points[3], points[0]],
			[points[0], points[2], points[4], points[1]],
			[points[3], points[6], points[7], points[5]]
		]

		
		#ax = fig.add_subplot(111, projection='3d')
		edgecolors = 'k' if showEdges else (0,0,0,0)
		faces = Poly3DCollection(edges, linewidths=1, edgecolors=edgecolors)
		faces.set_facecolor(color)

		ax.add_collection3d(faces)

		# Plot the points themselves to force the scaling of the axes
		ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

		ax.set_aspect('auto')

	def plot_box(self, box, ax, color=None, showEdges=True):
		"""
		box : obj of type "Box"
		
		ax: a matplotlib 3d axis obj i.e. from:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
		"""
		dx, dy, dz = box.dx, box.dy, box.dz
		x,y,z = box.x, box.y, box.z
		
		cube_definition = [(x,y,z),  (x+dx,y,z), (x,y+dy,z), (x,y,z+dz)]

		#print (cube_definition)
		self.plot_parallelepiped(cube_definition, ax, color, showEdges)

