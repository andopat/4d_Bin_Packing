import numpy as np
import math, copy, time
from joblib import Parallel, delayed, parallel_backend

INF = 1e9+7

class Node:
    def __init__(self, prev, p):
        self.prev_node = prev
        self.next_nodes = {}
        self.terminated = False
        self.value = None
        self.reward = 0

        self.q = 0
        self.w = 0
        self.n = 0
        self.p = p

    def is_expanded(self):
        return len(self.next_nodes) > 0

    def is_terminated(self):
        return self.terminated

    def terminate(self):
        self.terminated = True
        self.p = 0

    def update(self, value):
        self.n += 1
        self.w += value
        # normal average
        self.q = self.w / self.n
        # moving average

    def get_u_value(self):
        u_value = self.p * np.sqrt(self.prev_node.n)/(self.n+1)
        return u_value

    def get_q_value(self):
        return self.q

    def choose_best(self, c=1):
        assert len(self.next_nodes) > 0
        max_value = -INF
        max_nodes = []

        for (action, node) in self.next_nodes.items():
            # print(action, node.n)
            if node.n > 0:
                advanced_q = node.get_q_value() - node.prev_node.get_q_value()
                cur_value = advanced_q + c * node.get_u_value()
            else:
                cur_value = 0.0 + c * node.get_u_value()

            if math.isclose(cur_value, max_value, rel_tol=1e-5):
                max_nodes.append((action, node))
            elif cur_value > max_value:
                max_value = cur_value
                max_nodes.clear()
                max_nodes.append((action, node))

        assert len(max_nodes) > 0, max_nodes
        idx = np.random.randint(0, len(max_nodes))
        return max_nodes[idx]

    def choose_best_x(self, c=1):

        actions = [a for a, _ in self.next_nodes.items()]
        nodes  = [node for _, node in self.next_nodes.items()]
        a_cx = [node.get_q_value() - node.prev_node.get_q_value() if node.n > 0 else c * node.get_u_value() for node in nodes]
        a_n = [(a_cx[i], actions[i], nodes[i]) for i in range(len(a_cx))]        
        a_n.sort(key = lambda x: x[0])

        return (a_n[-1][1], a_n[-1][2])


    def expand(self, **kwargs):
        pass


class MCTSNode(Node):

    def __init__(self, prev, p):
        super().__init__(prev, p)
        self.q = 0

    def expand(self, **kwargs):

        start = time.time()
        
        credit = kwargs.get('credit')
        sim_env = kwargs.get('sim_env')
        model =  kwargs.get("model")
        check_print = kwargs.get("check_print")

        assert sim_env is not None

        if credit is not None:
            assert credit <=1 and credit >=0
        else:
            credit = 1

        if check_print:
            print("\t\t\tset up 1 took", time.time()-start)

        start = time.time()
        current_obs = sim_env.current_obs
        current_box_mask = sim_env.current_box_mask
        keep_actions = np.where(current_box_mask == 1)[0]
        if check_print:
            print("\t\t\tset up 2 took", time.time()-start)

        # get possibilities using neural network
        start = time.time()
        _, policy, _, value = model.predict(current_obs, current_box_mask)
        if check_print:
            # print("\t\t\tpredict took", time.time()-start, np.sum(current_box_mask))
            print("\t\t\tpredict took", time.time()-start)

        # start = time.time()
        # for i in range(len(current_box_mask)):
        #     action = i
        #     if current_box_mask[i] == 1: # !!! still use mask !!!
        #         action_possibility = credit * policy[action] #+ (1-credit) * (1/valid_action_num)
        #         self.next_nodes[action] = MCTSNode(self, action_possibility)
        # if check_print:
        #     # print("\t\t\tmasking", time.time()-start, np.sum(current_box_mask), len(self.next_nodes))
        #     print("\t\t\tmasking", time.time()-start,  len(self.next_nodes))

        start = time.time()
        for action in keep_actions:
            action_possibility = credit * policy[action] #+ (1-credit) * (1/valid_action_num)
            self.next_nodes[action] = MCTSNode(self, action_possibility)
        if check_print:
            # print("\t\t\tmasking2x", time.time()-start, np.sum(current_box_mask), len(self.next_nodes))
            print("\t\t\tmasking2x", time.time()-start, len(self.next_nodes))


        # no give-up action, default action is '0'
        if len(self.next_nodes) == 0:
            self.next_nodes[0] = MCTSNode(self, 1)

        # if len(sim_env.boxSeqGenerator.box_list) >= 1:
        #     start = time.time()
        #     value = self.roll_out(model, sim_env, check_print)
        #     if check_print:       
        #         print("\t\t\trollout took", time.time() - start)

        self.value = value


    def roll_out(self, model, sim_env, check_print=False, gamma=1):
        assert sim_env is not None

        #Number of actions taken
        reward_stack = []
        value = None
        total_boxes_left = len(sim_env.boxSeqGenerator.box_list) -  sim_env.current_box_id  

        if check_print:
            print("\t\t\t\ttotal boxes left for rollout:", total_boxes_left)

        for i in range(total_boxes_left):
            if check_print:
                print("\t\t\t\t\tremaining:", total_boxes_left-i, sim_env.current_box.name)   
            
            start = time.time()        
            _, policy, _, value = model.predict(sim_env.current_obs, sim_env.current_box_mask)
            if check_print:
                print("\t\t\t\t\trollout predict took", time.time()- start)

            policy = policy.data.cpu().numpy()

            action_sample = np.random.choice(policy.shape[0], p=policy)
            reward, done, _ = sim_env.step(action_sample)

            if not done and i+1 < total_boxes_left:
                reward_stack.append(reward)
            if done:
                reward_stack.append(reward)
                value = 0
                break

        for i in range(len(reward_stack)-1, -1, -1):
            value = reward_stack[i] + gamma * value
        return value




