import os, sys, math, random, time, copy, argparse

import numpy as np
from collections import deque
from joblib import Parallel, delayed, parallel_backend

import pandas as pd

import torch
from torch.multiprocessing import Process, Pipe
import torch.multiprocessing as mp

import config

from model_arch.model import NNetWrapper
from model_arch.net import CNNPro
from mcts.monteCarlo import MCTree

from pack_env.box import Box
from pack_env.packingEnv import PackEnv
from pack_env.packingHeuristic import PackHeuristic

from pack_env.plot import Map


def pack_customer_order(proc_id, cust_order_df, model, all_customer_order_ids, plot_packing=False, plot_file=None, check_print=False):

	start = time.time()

	order_id = all_customer_order_ids[proc_id]
	cust_order_df = cust_order_df[cust_order_df["ORDER_ID"] == order_id]
	cust_order_df.reset_index(drop=True, inplace=True)

	num_items = cust_order_df["ORDER_QTY"].sum()
	orderID = "ORDER_ID_" + str(order_id)
	
	#######################################################
	##### preparing input list of items to be packed #####
	items_info = []

	item_num = 1
	for i in range(len(cust_order_df)):
		item_id = cust_order_df.loc[i, "ITEM_ID"]
		# in case multiple orders for same item
		for j in range(int(cust_order_df.loc[i, "ORDER_QTY"])):
			box_len = cust_order_df.loc[i, "UNIT_LENGTH (Inches)"]
			box_wid = cust_order_df.loc[i, "UNIT_WIDTH (Inches)"]
			box_ht = cust_order_df.loc[i, "UNIT_HEIGHT (Inches)"]
			box_wt = cust_order_df.loc[i, "UNIT_WEIGHT (LBs)"]
			
			box_lwh = [box_len, box_wid, box_ht]
			box_dims = copy.deepcopy(box_lwh)
			box_dims.sort(reverse=True) # for each item rotate it so that X-axis(largest arm), Z-axis(smallest arm)

			l, w, h = box_dims

			# item dimesions get rounded up, containers dim get rounded down when integrizing
			dx = math.ceil(l) 
			dy = math.ceil(w)
			dz = math.ceil(h)

			# item_name = "ITEM_" + str(item_num)
			# item_name = "ITEM_NUM" + str(item_num) + "ItemID_" + str(item_id) + "_Num_" + str(j+1)								
			item_name = "ItemID_" + str(item_id) + "_Num_" + str(j+1)
			item_num += 1

			items_info.append([l, w, h, dx, dy, dz, box_wt, item_name, orderID, box_lwh])

	# sort all items by their X-value..only needed if unique items > 1
	if len(cust_order_df) > 1:
		items_info.sort(key=lambda x: x[0])
		items_info = items_info[::-1]

	# make input list
	input_box_list_orig = [] # for heuristic
	input_box_list_int = [] # with integerized sides for reinfrocement learning model
	for i in range(len(items_info)):
		l, w, h, dx, dy, dz, item_wt, item_name, orderID, box_lwh = items_info[i]
		item_box_orig = Box(dx=l, dy=w, dz=h, wt=item_wt, name=item_name, parent_gen=orderID, orig_size=box_lwh, orig_intXY_sort_size=(l,w,h))
		input_box_list_orig.append(item_box_orig)
		
		item_box_int = Box(dx=dx, dy=dy, dz=dz, wt=item_wt, name=item_name, parent_gen=orderID, orig_size=box_lwh, orig_intXY_sort_size=(dx, dy, dz))
		input_box_list_int.append(item_box_int)		
	#######################################################

	if len(items_info) > 700:
		return [proc_id, time.time() - start, 0, "[]", "Currently packing only <=700 items in one order"]


	######################################################
	############### Packing #############################
	try:
		# check heuristic - generally much faster eg .001-.05s/item in many trivial cases
		px = PackHeuristic(items_info, input_box_list_orig)
		packed, used_container_ids, used_container_names, packing_info, container_wise_packing = px.check_packing()
		time_taken = time.time() - start

		if packing_info == "can't be packed":
			# in case some LWH or wt violation for any item of the customer order
			return [proc_id, time_taken, 0, "[]", "Can't be packed - some item dim/wt violation"]


		ib = input_box_list_orig
		pck = packing_info
		pltf = plot_file

		# RL based packing.
		# for faster results avoid when:
		# (a) heuristic already gave best case solution eg 1 container 
		# (b) if there is just a single item type in a customer order, heuristics will suffice
		# (c) if too many items to pack, since RL can take ~0.25-1s/item and that much time is not available on client side eg for 10 items it can take upto 10s
		try:
			start = time.time()
			if (len(cust_order_df) > 1) and (len(items_info) <= 20) and (len(used_container_ids) > config.rl_threshold_num_containers):
				unique_container_ids = list(sorted(set(used_container_ids), key=used_container_ids.index))

				# print("using RL packing")
				packEnv = PackEnv(datagen_mode="predict", customer_order_list=input_box_list_int, init_container_ids_list=unique_container_ids)
				
				actionId_lookUp = packEnv.actionId_lookUp
				box_list = packEnv.boxSeqGenerator.box_list
				item_names = [b.name for b in packEnv.boxSeqGenerator.box_list]

				packEnv.reset(check_print=False, mode_mcts_sim=False)
				mctree = MCTree(model, packEnv, config.search_depth)

				for i in range(packEnv.boxSeqGenerator.num_boxes):
					# print("packing {}/{} name:{}".format(i, packEnv.boxSeqGenerator.num_boxes, packEnv.boxSeqGenerator.box_list[i].name))

					actionID = mctree.select_action(config.simulation_times, check_print=False)
					reward, done, info = packEnv.step(actionID, check_print=False, mode_mcts_sim=False)
					mctree.succeed(actionID)


				num_containers_new = len(packEnv.container_sets_status.container_placedBox_lookUp)

				if num_containers_new < len(used_container_ids):

					packing_info = packEnv.container_sets_status.container_placedBox_lookUp

					new_packing_info = {}
					cntrs = {}
					container_wise_packing = ""
					for key, packed in packing_info.items():
						cx = key.split("(")[0]
						if cx not in cntrs.keys():
							cntrs.update({cx:0})
						cntrs[cx] += 1
						new_cntr_name = cx + "(" + str(cntrs[cx]) + ")"
						new_packing_info[new_cntr_name] = packing_info[key]

						container_wise_packing += new_cntr_name + "<= #items:{} details =>".format(len(packed["packed_boxes"])) \
												+ str([b.basic_info() for b in packed["packed_boxes"]]) + "\n"

					container_wise_packing = container_wise_packing[:-1] # remove last newline
					used_container_names = [k for k in new_packing_info.keys()]
					used_container_ids = [packing["container_id"] for key, packing in new_packing_info.items()]

					time_taken = time.time() - start

					ib = input_box_list_int
					pck = new_packing_info
					pltf = plot_file[:-4] + "_RL_.gif"
		except:
			pass

		if plot_packing:
			x = Map(ib, pck, pltf)

		return [proc_id, time_taken, len(used_container_names), str(used_container_names), container_wise_packing]
		
	except Exception as e:
		print(e)
		return [proc_id, np.nan, 0, "[]", "Some unexpected error"]



if __name__=="__main__":

	torch.multiprocessing.set_start_method('spawn')

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='demo', help='demo | live; files in ./demo/ or ./live_predictions/; ')	
	parser.add_argument('--inputfile', default='single_customer_order_10_different_items.xlsx', help='customer order file, should be inside ./demo/input_files/ or ./live_predictions/input_files/ as from mode')
	parser.add_argument('--plot_packing', default=True, help='plot container-wise packing')

	args = parser.parse_args()
	customer_order_file = args.inputfile

	if args.mode == "live":
		inputdir = "./live_predictions"
	else:
		inputdir = "./demo"

	save_results_file_name = inputdir + "/pack_results/" + customer_order_file.split(".")[0] + "_packing.xlsx"
	save_plot_gif_file = inputdir + "/pack_results/" + customer_order_file.split(".")[0] + "_plot.gif"        

	cust_order_df = pd.read_excel(inputdir + "/input_files/" + customer_order_file, header=0)
	item_master_df = pd.read_excel(config.ITEM_MASTER_FILE, header=0)

	cust_order_df = cust_order_df.merge(item_master_df, on=["ITEM_ID"], how="left")

	cust_order_df["ORDER_ID"] = cust_order_df["ORDER_ID"].astype(str)
	unique_cust_order_ids = cust_order_df["ORDER_ID"].unique().tolist()

	print("# customer orders", len(unique_cust_order_ids), unique_cust_order_ids[:3], unique_cust_order_ids[-3:])

	if args.plot_packing and len(unique_cust_order_ids) == 1:
		args.plot_packing = True
	else:
		args.plot_packing = False

	start = time.time()
	net = CNNPro()
	net.to(torch.device("cpu")) # for multi-processing
	model = NNetWrapper(net)
	model.load_checkpoint(folder=config.epoch_dir, filename=config.save_model_name)


	print("generating predictions")
	start = time.time()
	with parallel_backend('multiprocessing', n_jobs=config.N_PARALLEL_JOBS):
		list_results = Parallel()(delayed(pack_customer_order)(i, cust_order_df, model, unique_cust_order_ids, args.plot_packing, save_plot_gif_file) for i in range(len(unique_cust_order_ids)))
	print("finished all packing in", time.time() - start)
	list_results.sort(key=lambda x:x[0])

	for i in range(len(unique_cust_order_ids)):
		first_row = cust_order_df.index[cust_order_df["ORDER_ID"] == unique_cust_order_ids[i]][0]
		cust_order_df.loc[first_row, "time_taken"] = list_results[i][1]
		cust_order_df.loc[first_row, "num_containers"] = list_results[i][2]
		cust_order_df.loc[first_row, "used_containers"] = str(list_results[i][3])
		cust_order_df.loc[first_row, "Container-wise packing-info"] = list_results[i][4]
		
	extra_cols = ["ITEM_NAME","DESCRIPTION", "UNIT_HEIGHT (Inches)", "UNIT_LENGTH (Inches)", "UNIT_WIDTH (Inches)", "UNIT_WEIGHT (LBs)", "UNIT_VOLUME (Cubic Feet)"]
	cust_order_df.drop(columns=extra_cols, inplace=True)

	cust_order_df.to_excel(save_results_file_name, index=False)



