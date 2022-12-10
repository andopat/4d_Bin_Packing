import copy, time
import numpy as np
import sys

from .node import MCTSNode

sys.path.append("../")
import config

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTree(object):
    def __init__(self, model, environment, search_depth=None, credit=1):        
        self.sim_env = environment
        self.root = MCTSNode(None, 1.0)
        self.model = model

        self.rollout_length = -1
        self.credit = credit
        self.c = 1

        self.subrt = 0
        self.reached_depth = -1

        if search_depth is not None:
            self.max_depth = search_depth
        else:
            self.max_depth = len(self.known_size_seq)-1


    def tree_policy(self, check_print=False):
        cur_node = self.root
        cur_depth = 0
        sim2_env = copy.deepcopy(self.sim_env)

        while True:
            # Terminated: back up
            if cur_node.is_terminated():
                # without future
                value = 0
                break

            # Not Expanded: expand node
            if not cur_node.is_expanded():
                start = time.time()
                pointer = cur_depth
                cur_node.expand(model=self.model,
                                credit=self.credit,
                                sim_env=sim2_env,
                                check_print=check_print)

                if check_print:
                    print("\t\texpand took", time.time() - start)

                value = cur_node.value
                break
            # reached max depth: back up
            if cur_depth == self.max_depth:
                value = cur_node.value
                break
            
            # not leaf node: use tree policy
            start = time.time()
            cur_action, next_node = cur_node.choose_best(self.c)
            if check_print:
                print("\t\tchoose best took", time.time() - start, cur_action)


            # Simulate time: take the action
            start = time.time()
            action_idx = cur_action
            reward, done, _ = sim2_env.step(action_idx)
            if check_print:
                print("\t\taction step took", time.time() - start)
                
            if check_print:
                print("\n\t\t\tselected", currentBox)

            next_node.reward = reward
            if done:
                self.subrt += 1
                if not next_node.is_terminated():
                    next_node.terminate()
                cur_node = next_node
                value = 0
                break
            cur_node = next_node
            cur_depth += 1


        if cur_depth > self.reached_depth:
            self.reached_depth = cur_depth
        start = time.time()
        self.backup(cur_node, value)
        # print("\tbackup took", time.time() - start)

    def backup(self, leaf_node, value, gamma=1):
        cur_node = leaf_node
        while True:
            value = cur_node.reward + gamma * value
            cur_node.update(value)
            if cur_node.prev_node is not None:
                cur_node = cur_node.prev_node
                continue
            break


    def select_action(self, sim_times, check_print=False, print_sim = False):
        check_print = not True
        print_sim = not True

        if check_print:
            print("\n**Selecting action for:", self.sim_env.current_box_id, self.sim_env.current_box.name)
        
        start1 = time.clock()
        for i in range(sim_times):
            start = time.time()

            if check_print and print_sim:
                print('\tsimulation',i+1)

            self.tree_policy(check_print)

            if check_print and print_sim:
                print("\tsimulation:", i+1, "finished in =>", time.time() - start)

        selcted_actionID, _ = self.root.choose_best(self.c)

        end = time.clock()
        # print("terminated node:", self.subrt)
        # print('reached depth:', self.reached_depth)
        # print('cost time', end-start1)   
        return selcted_actionID



    def succeed(self, put_action):
        put_action = int(put_action)
        new_node = self.root.next_nodes.get(put_action)
        assert new_node is not None
        new_node.p = 1.0
        new_node.prev_node = None
        self.root = new_node
        self.root.next_nodes = {}
        self.reached_depth = -1
        self.subrt = 0
