import copy 

UNIT_LWH = 1

# original container dimensions data from ./Data/ContainerMaster.xlsx
ORIGINAL_CONTAINERS_INFO = {
	0:{"name":  "BOX-012", "max_weight": 30, "length": 12,   "width": 12,  "height": 12},
	1:{"name":  "BOX-014", "max_weight": 30, "length": 5.5,  "width": 9,   "height": 12},
	2:{"name":  "BOX-018", "max_weight": 30, "length": 7.5,  "width": 13,  "height": 18},
	3:{"name":  "BOX-024", "max_weight": 30, "length": 9.75, "width": 18,  "height": 24},
	4:{"name":  "BOX-028", "max_weight": 30, "length": 16,   "width": 20,  "height": 28},
	5:{"name":  "BOX-075", "max_weight": 30, "length": 15,   "width": 18,  "height": 24},
	6:{"name":  "BOX-32C", "max_weight": 30, "length": 41,   "width": 24,  "height": 8},
	7:{"name":  "BOX-540", "max_weight": 30, "length": 11,   "width": 13,  "height": 21},
	8:{"name":  "BOX-681", "max_weight": 30, "length": 10,   "width": 20,  "height": 20},
	9:{"name":  "BOX-GOX", "max_weight": 30, "length": 24,   "width": 14,  "height": 8},
	10:{"name": "BOX-LRG", "max_weight": 30, "length": 3,    "width": 13,  "height": 18},
	11:{"name": "BOX-MED", "max_weight": 30, "length": 3,    "width": 11,  "height": 16},
	12:{"name": "BOX-SML", "max_weight": 30, "length": 2,    "width": 11,  "height": 13},
	13:{"name": "BOX-TUB", "max_weight": 10, "length": 6,    "width": 6,   "height": 36},
	14:{"name": "Box-GOH", "max_weight": 50, "length": 48,   "width": 20,  "height": 24},
	15:{"name": "ENV-001", "max_weight": 1,  "length": 7.25, "width": 0.5, "height": 12},
	16:{"name": "ENV-005", "max_weight": 1,  "length": 10.5, "width": 0.5, "height": 15},
	17:{"name": "ENV-007", "max_weight": 1,  "length": 14.25,"width": 0.5, "height": 20}
}


CONTAINERS_CONFIG = {"num_containers":len(ORIGINAL_CONTAINERS_INFO), "container_details":{}}

X, Y, Z, W, V = [], [], [], [], []
for key, value in ORIGINAL_CONTAINERS_INFO.items():

	name = value["name"]
	max_weight = value["max_weight"]
	
	# keep original size for heuristics which can handle real values
	l = value["length"]
	w = value["width"]
	h = value["height"]
	lwh = [l, w, h]
	lwh.sort(reverse=True) # X-axis(largest), Z-axis(smallest)
	Vol = l*w*h # in cubic inches
	l, w, h = lwh

	# integrize XY-grid for faster mapping of z-occupancyavailability
	xyz = []
	for v in lwh:
		if v >= 1:
			v = int(int(v)/UNIT_LWH)
		else:
			v = int(v/UNIT_LWH) # for 0.5-1
		xyz.append(v)
	xyz.sort(reverse=True)
	x, y, z = xyz # integrized in inches

	CONTAINERS_CONFIG["container_details"][key] = {"name":name, "max_weight":max_weight, "max_vol":Vol, "L":l, "W":w, "H":h, "X":x, "Y":y, "Z":z}

	X.append(x)
	Y.append(y)
	Z.append(z)	

	W.append(value["max_weight"] )
	V.append([key, Vol])

max_vol = max([V[i][1] for i in range(len(V))])
CONTAINERS_CONFIG.update({"min_X":min(X), "max_X":max(X), "min_Y":min(Y), "max_Y":max(Y), "min_Z":min(Z), "max_Z":max(Z), "max_W":max(W), "max_Vol":max_vol})
# print(CONTAINERS_CONFIG)
# input("check-containers-config")
