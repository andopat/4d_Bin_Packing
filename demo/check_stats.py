import pandas as pd


##### Containerization using our approach ####
# filename = "./pack_results/7778_customer_orders_total_14915_items_packing.xlsx"
filename = "./pack_results/OrderData_packing.xlsx"

df = pd.read_excel(filename, header=0)

print(df["ORDER_ID"].nunique(), df["ORDER_QTY"].sum())

df = df[df["num_containers"] != 0]
df.reset_index(drop=True, inplace=True)

print(df["ORDER_ID"].nunique(), df["ORDER_QTY"].sum(), df["num_containers"].sum())

orders = df["ORDER_ID"].unique()
#####

##### Previous containerization
dfx = pd.read_excel("../Data/ContainerizationOutput.xlsx", header=0)
dfx = dfx[dfx["ORDER_ID"].isin(orders)]
dfx.reset_index(drop=True,inplace=True)
# print(dfx["LPN_ID"].nunique())
######


##### Compare packing #####
df_stats = pd.DataFrame(data=[], columns=["ORDER_ID", "TOTAL_NUM_ITEMS", "NUM_CONTAINERS_NEW_METHOD", "NUM_CONTAINERS_OLD_METHOD"])

for i in range(len(orders)):
	order_id = orders[i]

	df_n = df[df["ORDER_ID"] == order_id].copy()
	df_n.reset_index(drop=True, inplace=True)
	num_new_containers = df_n.loc[0, "num_containers"]

	dfx_n = dfx[dfx["ORDER_ID"] == order_id].copy()
	dfx_n.reset_index(drop=True, inplace=True)
	# num_old_containers = len(dfx_n)
	num_old_containers = dfx_n["LPN_ID"].nunique()

	df_stats.loc[i, "ORDER_ID"] = order_id
	df_stats.loc[i, "TOTAL_NUM_ITEMS"] = df_n["ORDER_QTY"].sum()
	df_stats.loc[i, "NUM_CONTAINERS_NEW_METHOD"] = num_new_containers
	df_stats.loc[i, "NUM_CONTAINERS_OLD_METHOD"] = num_old_containers
	
df_stats.to_excel("./stats.xlsx", index=False)


# overall improvement - client's old vs our
total_contaiers_old = df_stats["NUM_CONTAINERS_OLD_METHOD"].sum()
total_contaiers_new = df_stats["NUM_CONTAINERS_NEW_METHOD"].sum()

improvement = (total_contaiers_old - total_contaiers_new) / total_contaiers_old *100
print(improvement)


# improvement for non-trivial cases eg when client's old method uses >=2 containers
df_stats_res = df_stats[df_stats["NUM_CONTAINERS_OLD_METHOD"]>=2]
df_stats_res.reset_index(drop=True, inplace=True)

total_contaiers_old = df_stats_res["NUM_CONTAINERS_OLD_METHOD"].sum()
total_contaiers_new = df_stats_res["NUM_CONTAINERS_NEW_METHOD"].sum()

res_improvement = (total_contaiers_old - total_contaiers_new) / total_contaiers_old *100
print(res_improvement)

df_stats_res.to_excel("./stats_nt.xlsx", index=False)
#####
